impl AprTransformer {

    /// Create a new APR transformer with the given configuration
    pub fn new(config: AprTransformerConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let intermediate_dim = config.intermediate_dim;

        let layers = (0..config.num_layers)
            .map(|_| AprTransformerLayer::empty(hidden_dim, intermediate_dim))
            .collect();

        Self {
            config,
            token_embedding: vec![0.0; vocab_size * hidden_dim],
            layers,
            output_norm_weight: vec![1.0; hidden_dim],
            output_norm_bias: None,
            lm_head_weight: vec![0.0; hidden_dim * vocab_size],
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    /// Get the model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.config
    }

    /// Generate tokens autoregressively (simplified version without KV cache)
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `max_tokens` - Maximum tokens to generate
    ///
    /// # Returns
    ///
    /// Generated token sequence (including prompt)
    pub fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_tokens {
            let logits = self.forward(&tokens)?;

            // Greedy sampling: take argmax
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32);

            tokens.push(next_token);

            // GH-330: Stop at EOS from model config (Design by Contract)
            if next_token == 0 {
                break;
            }
            if let Some(eos) = self.config.eos_token_id {
                if next_token == eos {
                    break;
                }
            }
        }

        Ok(tokens)
    }

    /// Get total number of parameters
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.token_embedding.len();
        for layer in &self.layers {
            count += layer.num_parameters();
        }
        count += self.output_norm_weight.len();
        count += self.output_norm_bias.as_ref().map_or(0, Vec::len);
        count += self.lm_head_weight.len();
        count += self.lm_head_bias.as_ref().map_or(0, Vec::len);
        count
    }

    /// Get memory size in bytes (F32 = 4 bytes per param)
    #[must_use]
    pub fn memory_size(&self) -> usize {
        self.num_parameters() * 4
    }

    /// Look up token embeddings
    #[must_use]
    pub fn embed(&self, token_ids: &[u32]) -> Vec<f32> {
        let hidden_dim = self.config.hidden_dim;
        let debug = std::env::var("REALIZE_DEBUG").is_ok();
        let mut embeddings = Vec::with_capacity(token_ids.len() * hidden_dim);

        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                if debug && token_id < 10 {
                    eprintln!(
                        "[DEBUG] embed token {}: offset={}, first 5: {:?}",
                        token_id,
                        offset,
                        &self.token_embedding[offset..offset + 5.min(hidden_dim)]
                    );
                }
                embeddings.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                // N-09: OOB token → zeros. Contract: embedding-lookup-v1.yaml
                eprintln!(
                    "Warning: AprTransformer::embed token_id {} OOB (offset={offset}, len={}). N-09 escape.",
                    token_id, self.token_embedding.len()
                );
                embeddings.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        embeddings
    }

    /// RMSNorm (delegates to helpers module)
    fn layer_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        helpers::rms_norm(input, weight, bias, self.config.hidden_dim, eps)
    }

    /// Matrix multiplication (delegates to helpers module)
    #[allow(clippy::unused_self)]
    fn matmul(&self, input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        helpers::f32_matmul(input, weight, in_dim, out_dim)
    }

    /// Add bias in-place (delegates to helpers module)
    #[allow(clippy::unused_self)]
    fn add_bias(&self, data: &mut [f32], bias: &[f32]) {
        helpers::add_bias_inplace(data, bias);
    }

    /// GELU activation (delegates to helpers module)
    #[allow(clippy::unused_self)]
    fn gelu(&self, data: &mut [f32]) {
        helpers::gelu_inplace(data);
    }

    /// Apply RoPE (delegates to helpers module)
    fn apply_rope_f32(&self, x: &mut [f32], position: usize, num_heads: usize, head_dim: usize) {
        helpers::apply_rope_f32(x, position, num_heads, head_dim, self.config.rope_theta);
    }
}
