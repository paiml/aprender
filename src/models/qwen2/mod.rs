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
    /// Uses 1-element tensor instead of vocab_size * hidden_size.
    ///
    /// **IMPORTANT**: This layer will NOT work for inference until
    /// `set_weight()` is called with real weights.
    pub fn placeholder(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            weight: Tensor::new(&[0.0], &[1]),
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
#[allow(clippy::struct_field_names)] // Standard ML naming convention
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

    /// Create a placeholder MLP with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
    pub fn placeholder(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::placeholder(hidden_size, intermediate_size),
            up_proj: Linear::placeholder(hidden_size, intermediate_size),
            down_proj: Linear::placeholder(intermediate_size, hidden_size),
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

    /// Create a placeholder decoder layer with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
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

    /// Forward pass with detailed profiling output.
    pub fn forward_profiled(
        &self,
        hidden_states: &Tensor,
        _position_ids: &[usize],
        _rope: &RotaryPositionEmbedding,
        _attention_mask: Option<&Tensor>,
    ) -> (Tensor, std::time::Duration, std::time::Duration) {
        use std::time::Instant;

        // Self-attention with pre-norm
        let residual = hidden_states.clone();
        let hidden = self.input_layernorm.forward(hidden_states);

        let attn_start = Instant::now();
        let (attn_output, _attn_weights) = self.self_attn.forward_self(&hidden, None);
        let attn_time = attn_start.elapsed();

        let hidden = add_tensors(&residual, &attn_output);

        // MLP with pre-norm
        let residual = hidden.clone();
        let hidden = self.post_attention_layernorm.forward(&hidden);

        let mlp_start = Instant::now();
        let mlp_output = self.mlp.forward(&hidden);
        let mlp_time = mlp_start.elapsed();

        (add_tensors(&residual, &mlp_output), attn_time, mlp_time)
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

    /// Create an uninitialized Qwen2 model with minimal memory allocation.
    ///
    /// This uses placeholder tensors (1 element each) instead of full weight
    /// matrices, reducing memory from ~2.5GB to ~1KB for Qwen2-0.5B.
    ///
    /// **IMPORTANT**: This model will NOT work for inference until weights
    /// are loaded via `load_from_apr()` or `load_from_safetensors()`.
    ///
    /// Per Native Library Mandate (Spec §2.4): Use this constructor when
    /// loading pre-trained weights to avoid OOM from double allocation.
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

    /// Forward with detailed profiling output.
    /// Prints timing breakdown for each component.
    pub fn forward_profiled(&mut self, input_ids: &[u32], position_ids: &[usize]) -> Tensor {
        use std::time::Instant;

        let total_start = Instant::now();

        // Embed tokens
        let embed_start = Instant::now();
        let mut hidden = self.embed_tokens.forward(input_ids);
        let embed_time = embed_start.elapsed();

        // Generate causal mask
        let seq_len = input_ids.len();
        let attention_mask = generate_causal_mask(seq_len);

        // Pass through decoder layers with profiling
        let mut total_attn = std::time::Duration::ZERO;
        let mut total_mlp = std::time::Duration::ZERO;

        let layers_start = Instant::now();
        for layer in &self.layers {
            let (output, attn_time, mlp_time) =
                layer.forward_profiled(&hidden, position_ids, &self.rope, Some(&attention_mask));
            hidden = output;
            total_attn += attn_time;
            total_mlp += mlp_time;
        }
        let layers_time = layers_start.elapsed();

        // Final normalization
        let norm_start = Instant::now();
        hidden = self.norm.forward(&hidden);
        let norm_time = norm_start.elapsed();

        // Project to vocabulary
        let lm_head_start = Instant::now();
        let output = self.lm_head.forward(&hidden);
        let lm_head_time = lm_head_start.elapsed();

        let total_time = total_start.elapsed();

        // Print profiling results
        eprintln!("\n=== Forward Pass Profile (seq_len={seq_len}) ===");
        eprintln!(
            "  Embedding:     {:>8.2}ms ({:>5.1}%)",
            embed_time.as_secs_f64() * 1000.0,
            embed_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        eprintln!(
            "  Layers total:  {:>8.2}ms ({:>5.1}%)",
            layers_time.as_secs_f64() * 1000.0,
            layers_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        eprintln!(
            "    - Attention: {:>8.2}ms ({:>5.1}%)",
            total_attn.as_secs_f64() * 1000.0,
            total_attn.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        eprintln!(
            "    - MLP:       {:>8.2}ms ({:>5.1}%)",
            total_mlp.as_secs_f64() * 1000.0,
            total_mlp.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        eprintln!(
            "  Final norm:    {:>8.2}ms ({:>5.1}%)",
            norm_time.as_secs_f64() * 1000.0,
            norm_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        eprintln!(
            "  LM head:       {:>8.2}ms ({:>5.1}%)",
            lm_head_time.as_secs_f64() * 1000.0,
            lm_head_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
        );
        eprintln!(
            "  TOTAL:         {:>8.2}ms",
            total_time.as_secs_f64() * 1000.0
        );
        eprintln!("==========================================\n");

        output
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
        self.generate_internal(prompt_ids, max_new_tokens, temperature, false)
    }

    /// Generate with profiling output (prints timing breakdown).
    pub fn generate_profiled(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Vec<u32> {
        self.generate_internal(prompt_ids, max_new_tokens, temperature, true)
    }

    fn generate_internal(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        profile: bool,
    ) -> Vec<u32> {
        let mut output_ids = prompt_ids.to_vec();

        for i in 0..max_new_tokens {
            let position_ids: Vec<usize> = (0..output_ids.len()).collect();
            // Only profile first token to avoid spam
            let logits = if profile && i == 0 {
                self.forward_profiled(&output_ids, &position_ids)
            } else {
                self.forward(&output_ids, &position_ids)
            };

            // Get last token logits
            let vocab_size = self.config.vocab_size;
            let last_pos = output_ids.len() - 1;
            let logits_slice = &logits.data()[last_pos * vocab_size..(last_pos + 1) * vocab_size];

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

    /// Get reference to language model head (for testing/inspection).
    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
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

    /// Load model from SafeTensors file.
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

/// SiLU (Swish) activation: x * sigmoid(x)
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
        .map_or(0, |(i, _)| i)
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

    // ========== Additional Coverage Tests ==========

    #[test]
    fn test_embedding_placeholder() {
        let emb = Embedding::placeholder(1000, 64);
        assert_eq!(emb.vocab_size, 1000);
        assert_eq!(emb.hidden_size, 64);
        // Placeholder has minimal weight
        assert_eq!(emb.weight.data().len(), 1);
    }

    #[test]
    fn test_embedding_set_weight() {
        let mut emb = Embedding::placeholder(10, 4);
        let new_weight = Tensor::ones(&[10, 4]);
        emb.set_weight(new_weight);
        assert_eq!(emb.weight().data().len(), 40);
    }

    #[test]
    fn test_embedding_weight_accessor() {
        let emb = Embedding::new(10, 4);
        let weight = emb.weight();
        assert_eq!(weight.shape(), &[10, 4]);
    }

    #[test]
    fn test_embedding_out_of_vocab() {
        let emb = Embedding::new(10, 4);
        // Token 100 is out of vocabulary (vocab_size=10)
        let output = emb.forward(&[0, 100, 2]);
        // Should still produce output (OOV token gets zeros)
        assert_eq!(output.shape(), &[1, 3, 4]);
    }

    #[test]
    fn test_qwen2_mlp_placeholder() {
        let _mlp = Qwen2MLP::placeholder(64, 128);
        // Placeholder MLPs should exist but have minimal weights
        // Note: Cannot do forward pass on placeholder (no weights set)
    }

    #[test]
    fn test_qwen2_mlp_mut_accessors() {
        let mut mlp = Qwen2MLP::new(64, 128);
        let gate = mlp.gate_proj_mut();
        assert!(gate.weight().shape().len() > 0);
        let up = mlp.up_proj_mut();
        assert!(up.weight().shape().len() > 0);
        let down = mlp.down_proj_mut();
        assert!(down.weight().shape().len() > 0);
    }

    #[test]
    fn test_qwen2_decoder_layer_placeholder() {
        let config = create_tiny_config();
        let _layer = Qwen2DecoderLayer::placeholder(&config);
        // Just verify placeholder can be created without panic
    }

    #[test]
    fn test_qwen2_decoder_layer_mut_accessors() {
        let config = create_tiny_config();
        let mut layer = Qwen2DecoderLayer::new(&config);

        let _attn = layer.self_attn_mut();
        let _mlp = layer.mlp_mut();
        let _input_norm = layer.input_layernorm_mut();
        let _post_norm = layer.post_attention_layernorm_mut();
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(4);
        assert_eq!(cache.keys.len(), 4);
        assert_eq!(cache.values.len(), 4);
        assert_eq!(cache.cached_len, 0);
        assert!(cache.keys.iter().all(|k| k.is_none()));
        assert!(cache.values.iter().all(|v| v.is_none()));
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KVCache::new(2);
        cache.cached_len = 10;
        cache.keys[0] = Some(Tensor::ones(&[1, 2, 3]));
        cache.values[0] = Some(Tensor::ones(&[1, 2, 3]));

        cache.clear();

        assert_eq!(cache.cached_len, 0);
        assert!(cache.keys.iter().all(|k| k.is_none()));
        assert!(cache.values.iter().all(|v| v.is_none()));
    }

    #[test]
    fn test_qwen2_model_uninitialized() {
        let config = create_tiny_config();
        let model = Qwen2Model::new_uninitialized(&config);
        assert_eq!(model.num_layers(), 2);
        // Uninitialized model has placeholder weights
    }

    #[test]
    fn test_qwen2_model_train_eval() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);

        model.train();
        assert!(model.training);

        model.eval();
        assert!(!model.training);
    }

    #[test]
    fn test_qwen2_model_cache_operations() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);

        // Initially no cache
        assert!(model.kv_cache.is_none());

        // Enable cache
        model.enable_cache();
        assert!(model.kv_cache.is_some());

        // Clear cache (should not panic even if empty)
        model.clear_cache();

        // Disable cache
        model.disable_cache();
        assert!(model.kv_cache.is_none());

        // Clear on disabled cache should not panic
        model.clear_cache();
    }

    #[test]
    fn test_qwen2_model_weight_names() {
        let config = create_tiny_config();
        let model = Qwen2Model::new(&config);

        let names = model.weight_names();
        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
        assert!(names.contains(&"lm_head.weight".to_string()));
        // Should have layer-specific names
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"model.layers.1.mlp.gate_proj.weight".to_string()));
    }

    #[test]
    fn test_qwen2_model_weight_info() {
        let config = create_tiny_config();
        let model = Qwen2Model::new(&config);

        let info = model.weight_info();
        assert!(info.contains_key("model.embed_tokens.weight"));
        assert_eq!(info["model.embed_tokens.weight"], vec![1000, 64]);
        assert!(info.contains_key("model.norm.weight"));
        assert_eq!(info["model.norm.weight"], vec![64]);
    }

    #[test]
    fn test_qwen2_model_weights() {
        let config = create_tiny_config();
        let model = Qwen2Model::new(&config);

        let weights = model.weights();
        assert!(weights.contains_key("model.embed_tokens.weight"));
        assert_eq!(weights["model.embed_tokens.weight"].len(), 1000 * 64);
    }

    #[test]
    fn test_qwen2_model_num_parameters() {
        let config = create_tiny_config();
        let model = Qwen2Model::new(&config);

        let num_params = model.num_parameters();
        // Should have embedding + layers + norm + lm_head
        assert!(num_params > 0);
        // Embedding alone is 1000 * 64 = 64000
        assert!(num_params >= 64000);
    }

    #[test]
    fn test_qwen2_model_mut_accessors() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);

        let _embed = model.embed_tokens_mut();
        let layer = model.layer_mut(0);
        assert!(layer.is_some());
        let bad_layer = model.layer_mut(100);
        assert!(bad_layer.is_none());
        let _norm = model.norm_mut();
        let _lm_head = model.lm_head_mut();
    }

    #[test]
    fn test_qwen2_model_generate_greedy() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);
        model.eval();

        let prompt = vec![1u32, 2, 3];
        // Generate with temperature=0 (greedy)
        let output = model.generate(&prompt, 2, 0.0, 1.0);

        // Should have prompt + new tokens
        assert!(output.len() >= prompt.len());
        assert!(output.len() <= prompt.len() + 2);
        // Prompt should be preserved
        assert_eq!(&output[..3], &[1, 2, 3]);
    }

    #[test]
    fn test_qwen2_model_generate_with_temperature() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);
        model.eval();

        let prompt = vec![1u32, 2, 3];
        // Generate with temperature > 0
        let output = model.generate(&prompt, 2, 0.8, 1.0);

        assert!(output.len() >= prompt.len());
    }

    #[test]
    fn test_sample_with_temperature() {
        let logits = vec![10.0f32, 1.0, 0.0, -1.0];

        // With low temperature, should mostly pick index 0
        let mut count_0 = 0;
        for _ in 0..10 {
            let sample = sample_with_temperature(&logits, 0.1);
            if sample == 0 {
                count_0 += 1;
            }
        }
        // With temperature 0.1, should heavily favor index 0
        assert!(count_0 >= 5, "Expected mostly 0s, got {count_0}/10");
    }

    #[test]
    fn test_sample_with_high_temperature() {
        let logits = vec![1.0f32, 1.0, 1.0, 1.0];

        // With uniform logits, all indices should be possible
        let mut seen = [false; 4];
        for _ in 0..100 {
            let sample = sample_with_temperature(&logits, 1.0) as usize;
            if sample < 4 {
                seen[sample] = true;
            }
        }
        // Should see at least some variety
        let variety = seen.iter().filter(|&&x| x).count();
        assert!(
            variety >= 2,
            "Expected variety, but only saw {variety} different values"
        );
    }

    #[test]
    fn test_elementwise_mul() {
        let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::new(&[2.0, 3.0, 4.0], &[3]);
        let c = elementwise_mul(&a, &b);
        assert_eq!(c.data(), &[2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_add_tensors() {
        let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::new(&[4.0, 5.0, 6.0], &[3]);
        let c = add_tensors(&a, &b);
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_argmax_empty() {
        let slice: [f32; 0] = [];
        // Should return 0 for empty slice
        assert_eq!(argmax(&slice), 0);
    }

    #[test]
    fn test_argmax_single() {
        let slice = [42.0f32];
        assert_eq!(argmax(&slice), 0);
    }

    #[test]
    fn test_causal_mask_size_1() {
        let mask = generate_causal_mask(1);
        assert_eq!(mask.shape(), &[1, 1]);
        assert_eq!(mask.data()[0], 0.0);
    }

    #[test]
    fn test_qwen2_decoder_layer_forward() {
        let config = create_tiny_config();
        let layer = Qwen2DecoderLayer::new(&config);
        let rope = RotaryPositionEmbedding::with_base(16, 128, 10000.0);

        let hidden = Tensor::ones(&[1, 5, 64]);
        let position_ids: Vec<usize> = (0..5).collect();

        let output = layer.forward(&hidden, &position_ids, &rope, None);
        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    // =========================================================================
    // Regression test: lm_head weight tying (GH-XXX)
    // =========================================================================

    #[test]
    fn test_linear_placeholder_not_ready() {
        // GIVEN: a placeholder Linear layer (simulating uninitialized model)
        let linear = Linear::placeholder(64, 128);

        // THEN: it should NOT be ready for inference
        assert!(
            !linear.is_ready(),
            "Placeholder Linear should not be ready (weight_t is None)"
        );
    }

    #[test]
    fn test_linear_after_set_weight_is_ready() {
        // GIVEN: a placeholder Linear layer
        let mut linear = Linear::placeholder(64, 128);
        assert!(!linear.is_ready(), "Precondition: placeholder not ready");

        // WHEN: set_weight is called
        let weight = Tensor::ones(&[128, 64]);
        linear.set_weight(weight);

        // THEN: it should be ready for inference
        assert!(
            linear.is_ready(),
            "Linear should be ready after set_weight (weight_t cached)"
        );
    }

    #[test]
    fn test_uninitialized_model_lm_head_not_ready() {
        // GIVEN: an uninitialized model (using placeholder constructors)
        let config = create_tiny_config();
        let model = Qwen2Model::new_uninitialized(&config);

        // THEN: lm_head should NOT be ready (no weights loaded)
        assert!(
            !model.lm_head().is_ready(),
            "Uninitialized model's lm_head should not be ready"
        );
    }

    /// Regression test for weight tying bug in load_from_safetensors.
    ///
    /// When SafeTensors file uses weight tying (lm_head shares weights with
    /// embed_tokens), there is no "lm_head.weight" tensor. The loader must
    /// fall back to using "model.embed_tokens.weight" for lm_head.
    ///
    /// Without this fix, lm_head.weight_t remains None and forward() panics.
    #[test]
    fn test_safetensors_weight_tying_lm_head_ready() {
        // This test verifies the INVARIANT: after load_from_safetensors,
        // ALL Linear layers must be ready (weight_t is Some).
        //
        // We test this by loading a real SafeTensors file if available,
        // or skip if not (integration test covers this).
        let safetensors_path = std::path::Path::new(
            "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/model.safetensors"
        );

        if !safetensors_path.exists() {
            // Skip if model not downloaded (CI may not have it)
            eprintln!("Skipping weight tying test: SafeTensors file not found");
            return;
        }

        // GIVEN: Qwen2 config matching the SafeTensors file
        let config = Qwen2Config::qwen2_0_5b_instruct();

        // WHEN: loading with new_uninitialized + load_from_safetensors
        let mut model = Qwen2Model::new_uninitialized(&config);
        let loaded = model
            .load_from_safetensors(safetensors_path)
            .expect("Should load SafeTensors");

        // THEN: lm_head must be ready (weight_t cached via weight tying)
        assert!(
            model.lm_head().is_ready(),
            "BUG: lm_head not ready after load_from_safetensors! \
             Weight tying fallback not implemented. Loaded {} tensors.",
            loaded
        );
    }

    // =========================================================================
    // Section S: Popperian Falsification Tests for Qwen2 Native Inference
    // =========================================================================
    //
    // These tests follow Karl Popper's criterion of demarcation: each test
    // specifies conditions under which the claim would be PROVEN FALSE.
    // A test that cannot fail is not scientific.
    //
    // Reference: Popper, K. (1959). The Logic of Scientific Discovery.
    // =========================================================================

    /// S1: Tokenizer loads from tokenizer.json
    /// FALSIFICATION: Encoding "Hello" returns empty or panics
    #[test]
    fn s1_tokenizer_loads_from_json() {
        let tokenizer_path = std::path::Path::new("/home/noah/.cache/qwen2/tokenizer.json");

        if !tokenizer_path.exists() {
            eprintln!("SKIP S1: tokenizer.json not found at {:?}", tokenizer_path);
            eprintln!("Download: curl -L -o ~/.cache/qwen2/tokenizer.json \\");
            eprintln!("  https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/tokenizer.json");
            return;
        }

        let json = std::fs::read_to_string(tokenizer_path).expect("read tokenizer.json");
        let tokenizer = crate::text::bpe::load_from_json(&json).expect("parse tokenizer.json");
        let tokens = tokenizer.encode("Hello");

        assert!(
            !tokens.is_empty(),
            "FALSIFIED S1: encode('Hello') returned empty. Tokenizer not functional."
        );

        println!("S1 PASSED: encode('Hello') -> {} tokens", tokens.len());
    }

    /// S2: Tokenizer round-trips ASCII correctly
    /// FALSIFICATION: decode(encode("Hello")) != "Hello"
    #[test]
    fn s2_tokenizer_roundtrip_ascii() {
        let tokenizer_path = std::path::Path::new("/home/noah/.cache/qwen2/tokenizer.json");

        if !tokenizer_path.exists() {
            eprintln!("SKIP S2: tokenizer.json not found");
            return;
        }

        let json = std::fs::read_to_string(tokenizer_path).expect("read");
        let tokenizer = crate::text::bpe::load_from_json(&json).expect("parse");

        let original = "Hello";
        let encoded = tokenizer.encode(original);
        let decoded = tokenizer.decode(&encoded);

        // Allow for whitespace normalization
        let decoded_trimmed = decoded.trim();
        assert!(
            decoded_trimmed == original || decoded.contains(original),
            "FALSIFIED S2: roundtrip failed. '{}' -> {:?} -> '{}'",
            original,
            encoded,
            decoded
        );

        println!("S2 PASSED: '{}' -> {:?} -> '{}'", original, encoded, decoded);
    }

    /// S3: Tokenizer handles Qwen2 special tokens
    /// FALSIFICATION: is_eos(151645) returns false
    #[test]
    fn s3_tokenizer_special_tokens() {
        use crate::text::bpe::Qwen2BpeTokenizer;

        let tokenizer = Qwen2BpeTokenizer::new();

        // <|im_end|> = 151645 is the EOS token
        assert!(
            tokenizer.is_eos(151645),
            "FALSIFIED S3: is_eos(151645) returned false. <|im_end|> not recognized."
        );

        // <|im_start|> = 151644 is the BOS token
        assert!(
            tokenizer.is_bos(151644),
            "FALSIFIED S3: is_bos(151644) returned false. <|im_start|> not recognized."
        );

        println!("S3 PASSED: Special tokens recognized correctly");
    }

    /// S4: Model loads from SafeTensors without OOM
    /// FALSIFICATION: OOM on 16GB machine OR load fails
    /// NOTE: Timing removed - use `cargo bench` for performance testing
    #[test]
    fn s4_model_loads_memory_efficient() {
        let safetensors_path = std::path::Path::new(
            "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/model.safetensors"
        );

        if !safetensors_path.exists() {
            eprintln!("SKIP S4: model.safetensors not found");
            return;
        }

        let config = Qwen2Config::qwen2_0_5b_instruct();
        let start = std::time::Instant::now();

        // Use memory-efficient loading
        let mut model = Qwen2Model::new_uninitialized(&config);
        let loaded = model.load_from_safetensors(safetensors_path);

        let elapsed = start.elapsed();

        assert!(
            loaded.is_ok(),
            "FALSIFIED S4: Model load failed: {:?}",
            loaded.err()
        );

        // Log timing for observability (no assertion - use benchmarks for perf)
        println!(
            "S4 PASSED: Loaded {} tensors in {:.2}s",
            loaded.unwrap_or(0),
            elapsed.as_secs_f32()
        );
    }

    /// S5: Model loads exactly 219 weight tensors
    /// FALSIFICATION: Tensor count != 219
    #[test]
    fn s5_model_tensor_count() {
        let safetensors_path = std::path::Path::new(
            "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/model.safetensors"
        );

        if !safetensors_path.exists() {
            eprintln!("SKIP S5: model.safetensors not found");
            return;
        }

        let config = Qwen2Config::qwen2_0_5b_instruct();
        let mut model = Qwen2Model::new_uninitialized(&config);
        let loaded = model
            .load_from_safetensors(safetensors_path)
            .expect("load");

        // Qwen2-0.5B has exactly 219 tensors:
        // - 1 embed_tokens
        // - 24 layers * 9 tensors (q,k,v,o,gate,up,down,input_norm,post_norm) = 216
        // - 1 final norm
        // - 1 lm_head (tied with embed_tokens)
        assert_eq!(
            loaded, 219,
            "FALSIFIED S5: Expected 219 tensors, got {}",
            loaded
        );

        println!("S5 PASSED: Loaded exactly 219 tensors");
    }

    /// S6: Embedding lookup returns correct shape
    /// FALSIFICATION: Output shape != [1, seq_len, 896]
    #[test]
    fn s6_embedding_shape() {
        let config = create_tiny_config();
        let emb = Embedding::new(1000, config.hidden_size);

        let input_ids = vec![1u32, 2, 3, 4, 5];
        let output = emb.forward(&input_ids);

        assert_eq!(
            output.shape(),
            &[1, 5, config.hidden_size],
            "FALSIFIED S6: Embedding shape {:?} != expected [1, 5, {}]",
            output.shape(),
            config.hidden_size
        );

        println!("S6 PASSED: Embedding shape correct");
    }

    /// S11: Logits shape matches vocab
    /// FALSIFICATION: Output shape != [1, seq_len, vocab_size]
    #[test]
    fn s11_logits_shape() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);

        let input_ids = vec![1u32, 2, 3];
        let position_ids: Vec<usize> = (0..3).collect();
        let logits = model.forward(&input_ids, &position_ids);

        assert_eq!(
            logits.shape(),
            &[1, 3, config.vocab_size],
            "FALSIFIED S11: Logits shape {:?} != expected [1, 3, {}]",
            logits.shape(),
            config.vocab_size
        );

        println!("S11 PASSED: Logits shape matches vocab");
    }

    /// S12: Logits are finite (no NaN/Inf)
    /// FALSIFICATION: Any NaN or Inf in output
    #[test]
    fn s12_logits_finite() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);

        let input_ids = vec![1u32, 2, 3];
        let position_ids: Vec<usize> = (0..3).collect();
        let logits = model.forward(&input_ids, &position_ids);

        let has_nan = logits.data().iter().any(|x| x.is_nan());
        let has_inf = logits.data().iter().any(|x| x.is_infinite());

        assert!(
            !has_nan,
            "FALSIFIED S12: Logits contain NaN values"
        );
        assert!(
            !has_inf,
            "FALSIFIED S12: Logits contain Inf values"
        );

        println!("S12 PASSED: All logits are finite");
    }

    /// S14: Top-1 token is deterministic (temp=0)
    /// FALSIFICATION: Same input produces different outputs
    #[test]
    fn s14_deterministic_generation() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);

        let input_ids = vec![1u32, 2, 3];

        // Generate twice with temperature=0 (greedy)
        let output1 = model.generate(&input_ids, 5, 0.0, 0.9);
        let output2 = model.generate(&input_ids, 5, 0.0, 0.9);

        assert_eq!(
            output1, output2,
            "FALSIFIED S14: Different outputs for same input with temp=0.\n  Run 1: {:?}\n  Run 2: {:?}",
            output1, output2
        );

        println!("S14 PASSED: Generation is deterministic at temp=0");
    }

    /// S20: Response length <= max_new_tokens
    /// FALSIFICATION: Output exceeds requested length
    #[test]
    fn s20_length_control() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);

        let input_ids = vec![1u32, 2, 3];
        let max_new_tokens = 10;

        let output = model.generate(&input_ids, max_new_tokens, 0.7, 0.9);
        let new_tokens = output.len() - input_ids.len();

        assert!(
            new_tokens <= max_new_tokens,
            "FALSIFIED S20: Generated {} tokens > max {}",
            new_tokens,
            max_new_tokens
        );

        println!("S20 PASSED: Generated {} <= {} tokens", new_tokens, max_new_tokens);
    }
}
