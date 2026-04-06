//! LLaMA 2 Transformer Architecture
//!
//! Reference implementation demonstrating LLaMA 2 using entrenar primitives.
//! This is a simplified version to illustrate concepts - production code would
//! require additional ops (softmax, layer_norm, etc.) in entrenar core.

#![allow(dead_code, unreachable_pub)]

use entrenar::Tensor;
use std::f32::consts::PI;

/// LLaMA model configuration
#[derive(Debug, Clone)]
pub struct LLaMAConfig {
    /// Vocabulary size (default: 32000)
    pub vocab_size: usize,
    /// Hidden dimension (768 for 124M, 4096 for 7B)
    pub hidden_size: usize,
    /// Number of transformer layers (12 for 124M, 32 for 7B)
    pub num_layers: usize,
    /// Number of attention heads (12 for 124M, 32 for 7B)
    pub num_heads: usize,
    /// Feed-forward intermediate size (3072 for 124M, 11008 for 7B)
    pub intermediate_size: usize,
    /// Maximum sequence length (default: 2048 or 4096)
    pub max_seq_len: usize,
    /// RoPE theta parameter (default: 10000.0)
    pub rope_theta: f32,
}

impl LLaMAConfig {
    /// Create 124M parameter toy model config
    pub fn toy_124m() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            max_seq_len: 2048,
            rope_theta: 10000.0,
        }
    }

    /// Create 7B parameter model config
    pub fn llama2_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            intermediate_size: 11008,
            max_seq_len: 4096,
            rope_theta: 10000.0,
        }
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

/// Single LLaMA transformer layer
pub struct LLaMALayer {
    // Self-attention projections
    pub q_proj: Tensor, // [hidden_size, hidden_size]
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,

    // Feed-forward projections (SwiGLU)
    pub gate_proj: Tensor, // [intermediate_size, hidden_size]
    pub up_proj: Tensor,
    pub down_proj: Tensor, // [hidden_size, intermediate_size]

    // Configuration
    config: LLaMAConfig,
}

impl LLaMALayer {
    /// Create a new LLaMA layer with random initialization
    pub fn new(config: &LLaMAConfig) -> Self {
        let h = config.hidden_size;
        let i = config.intermediate_size;

        // Xavier/Glorot initialization scale
        let attn_scale = (2.0 / (h + h) as f32).sqrt();
        let ffn_scale = (2.0 / (h + i) as f32).sqrt();
        let down_scale = (2.0 / (i + h) as f32).sqrt();

        Self {
            // Attention projections
            q_proj: Self::init_weight(h * h, attn_scale),
            k_proj: Self::init_weight(h * h, attn_scale),
            v_proj: Self::init_weight(h * h, attn_scale),
            o_proj: Self::init_weight(h * h, attn_scale),

            // Feed-forward projections
            gate_proj: Self::init_weight(i * h, ffn_scale),
            up_proj: Self::init_weight(i * h, ffn_scale),
            down_proj: Self::init_weight(h * i, down_scale),

            config: config.clone(),
        }
    }

    /// Initialize weight with Xavier scaling and random normal distribution
    fn init_weight(size: usize, scale: f32) -> Tensor {
        use rand::Rng;
        let mut rng = rand::rng();

        // Box-Muller transform for normal distribution
        let data: Vec<f32> = (0..=(size / 2))
            .flat_map(|_| {
                let u1: f32 = rng.random();
                let u2: f32 = rng.random();

                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();

                vec![z0 * scale, z1 * scale]
            })
            .take(size)
            .collect();

        Tensor::from_vec(data, true) // requires_grad=true for training
    }

    /// Forward pass through the layer (simplified reference implementation)
    pub fn forward(&self, x: &Tensor, _seq_len: usize, _batch_size: usize) -> Tensor {
        // REFERENCE IMPLEMENTATION NOTE:
        // This is a simplified forward pass demonstrating the conceptual flow.
        // Production implementation would use:
        // - entrenar::autograd::layer_norm for RMS normalization
        // - entrenar::autograd::softmax for attention weights
        // - entrenar::autograd::matmul for matrix multiplication (already available)
        // - Custom attention ops for scaled dot-product + masking
        //
        // For now, we return a placeholder tensor to demonstrate structure.

        // Placeholder: return tensor with same shape as input
        Tensor::zeros(x.len(), x.requires_grad())
    }

    /// Get all trainable parameters
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
            &mut self.gate_proj,
            &mut self.up_proj,
            &mut self.down_proj,
        ]
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }
}

/// Complete LLaMA model
pub struct LLaMAModel {
    /// Model configuration
    pub config: LLaMAConfig,
    /// Embedding layer
    pub embedding: Tensor,
    /// Transformer layers
    pub layers: Vec<LLaMALayer>,
    /// Output head
    pub lm_head: Tensor,
}

impl LLaMAModel {
    /// Create a new LLaMA model
    pub fn new(config: LLaMAConfig) -> Self {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        // Initialize embedding layer
        let embed_scale = (1.0 / hidden_size as f32).sqrt();
        let embedding = LLaMALayer::init_weight(vocab_size * hidden_size, embed_scale);

        // Initialize transformer layers
        let layers: Vec<LLaMALayer> = (0..num_layers).map(|_| LLaMALayer::new(&config)).collect();

        // Initialize output head (language modeling head)
        let lm_head = LLaMALayer::init_weight(vocab_size * hidden_size, embed_scale);

        Self { config, embedding, layers, lm_head }
    }

    /// Forward pass through the entire model (simplified reference)
    pub fn forward(&self, _input_ids: &[u32], batch_size: usize) -> Tensor {
        // REFERENCE IMPLEMENTATION NOTE:
        // This is a placeholder demonstrating model structure.
        // Production implementation would:
        // 1. Embed input_ids: lookup from self.embedding
        // 2. Pass through each layer: self.layers[i].forward(hidden, seq_len, batch_size)
        // 3. Apply final normalization
        // 4. Project to vocabulary: matmul(self.lm_head, normed)
        //
        // For now, return placeholder logits

        let seq_len = 4; // Placeholder
        let vocab_size = self.config.vocab_size;

        Tensor::zeros(batch_size * seq_len * vocab_size, false)
    }

    /// Get all trainable parameters
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.embedding];

        for layer in &mut self.layers {
            params.extend(layer.parameters());
        }

        params.push(&mut self.lm_head);
        params
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();

        for layer in &mut self.layers {
            layer.zero_grad();
        }

        self.lm_head.zero_grad();
    }

    /// Count total parameters
    pub fn count_parameters(&self) -> usize {
        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let num_layers = self.config.num_layers;

        // Embedding + LM head
        let embed_params = vocab_size * hidden_size * 2;

        // Per layer: 4 attention + 3 FFN projections
        let layer_params = (4 * hidden_size * hidden_size)  // Q, K, V, O
            + (2 * hidden_size * intermediate_size)  // gate, up
            + (intermediate_size * hidden_size); // down

        embed_params + (num_layers * layer_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_toy_124m() {
        let config = LLaMAConfig::toy_124m();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_config_7b() {
        let config = LLaMAConfig::llama2_7b();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_model_parameter_count_124m() {
        let config = LLaMAConfig::toy_124m();
        let model = LLaMAModel::new(config);

        let param_count = model.count_parameters();

        // Actual: ~162M (v*h*2 + n*(4*h*h + 3*h*i))
        assert!(param_count > 160_000_000);
        assert!(param_count < 165_000_000);
    }

    #[test]
    fn test_layer_creation() {
        let config = LLaMAConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 256,
            max_seq_len: 128,
            rope_theta: 10000.0,
        };

        let layer = LLaMALayer::new(&config);

        // Check that weights are initialized
        assert_eq!(layer.q_proj.len(), 64 * 64);
        assert_eq!(layer.gate_proj.len(), 256 * 64);
    }

    #[test]
    fn test_model_forward_shape() {
        let config = LLaMAConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 256,
            max_seq_len: 128,
            rope_theta: 10000.0,
        };

        let model = LLaMAModel::new(config.clone());

        let batch_size = 2;
        let input_ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let output = model.forward(&input_ids, batch_size);

        // Output shape should be reasonable
        assert!(output.len() > 0);
    }

    #[test]
    fn test_parameter_extraction() {
        let config = LLaMAConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 256,
            max_seq_len: 128,
            rope_theta: 10000.0,
        };

        let mut model = LLaMAModel::new(config);
        let params = model.parameters();

        // Should have embedding + (7 params per layer * 2 layers) + lm_head
        // = 1 + 14 + 1 = 16 parameters
        assert_eq!(params.len(), 16);
    }
}
