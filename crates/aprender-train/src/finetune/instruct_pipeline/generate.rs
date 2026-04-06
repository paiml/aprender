//! Text generation methods: `generate`, `generate_chat`.

#[allow(clippy::wildcard_imports)]
use super::*;

impl InstructPipeline {
    /// Autoregressive text generation with LoRA adapters (entrenar#246).
    ///
    /// Generates tokens one at a time using the transformer + LoRA forward pass.
    /// Supports greedy decoding (temperature=0) and temperature-scaled sampling
    /// with optional top-k filtering.
    ///
    /// # Arguments
    /// * `prompt` - Input text to continue from
    /// * `config` - Generation parameters (max tokens, temperature, top-k)
    ///
    /// # Returns
    /// Generated text (excluding the input prompt)
    ///
    /// # Errors
    /// Returns error if no tokenizer is loaded.
    pub fn generate(&self, prompt: &str, config: &GenerateConfig) -> crate::Result<String> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            crate::Error::ConfigError("No tokenizer loaded — cannot generate text".into())
        })?;

        let mut token_ids = tokenizer.encode(prompt);
        let prompt_len = token_ids.len();
        let eos_token = tokenizer.eos_id().unwrap_or(151643); // Qwen2 default EOS

        let vocab_size = self.model.config().vocab_size;

        for _ in 0..config.max_new_tokens {
            // Truncate to max_seq_len if needed
            if token_ids.len() >= self.config.max_seq_len {
                break;
            }

            // Forward pass with LoRA
            let hidden = self.model.forward_hidden_with_lora(&token_ids, &self.lora_layers);
            let seq_len = token_ids.len();
            let hidden_size = self.model.config().hidden_size;

            // Apply lm_head to get logits
            let lm_weight = self.model.lm_head_weight();
            let logits =
                crate::autograd::matmul_nt(&hidden, lm_weight, seq_len, hidden_size, vocab_size);

            // Extract logits for last position
            let logits_data = logits.data();
            let logits_slice = logits_data.as_slice().unwrap_or(&[]);
            let last_pos_start = (seq_len - 1) * vocab_size;
            let last_pos_logits = &logits_slice[last_pos_start..last_pos_start + vocab_size];

            // Sample next token
            let next_token = sample_token(last_pos_logits, config.temperature, config.top_k);

            if next_token == eos_token {
                break;
            }

            // Check stop tokens
            if config.stop_tokens.contains(&next_token) {
                break;
            }

            token_ids.push(next_token);
        }

        // Decode only the generated part (not the prompt)
        let generated_ids = &token_ids[prompt_len..];
        Ok(tokenizer.decode(generated_ids))
    }

    /// Generate a chat response using ChatML format (entrenar#246).
    ///
    /// Formats messages as ChatML (`<|im_start|>` / `<|im_end|>`) and generates
    /// the assistant's response.
    ///
    /// # Arguments
    /// * `system` - System prompt
    /// * `user_message` - User's input message
    /// * `config` - Generation parameters
    ///
    /// # Returns
    /// The assistant's generated response text.
    ///
    /// # Errors
    /// Returns error if no tokenizer is loaded.
    pub fn generate_chat(
        &self,
        system: &str,
        user_message: &str,
        config: &GenerateConfig,
    ) -> crate::Result<String> {
        let prompt = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user_message}<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        let mut response = self.generate(&prompt, config)?;

        // Strip trailing <|im_end|> if present
        if let Some(stripped) = response.strip_suffix("<|im_end|>") {
            response = stripped.to_string();
        }

        Ok(response)
    }
}

impl GenerateConfig {
    /// Create a greedy decoding config (deterministic, always picks highest probability token).
    #[must_use]
    pub fn greedy(max_new_tokens: usize) -> Self {
        contract_pre_greedy!();
        Self { max_new_tokens, temperature: 0.0, top_k: 0, stop_tokens: Vec::new() }
    }
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self { max_new_tokens: 256, temperature: 0.7, top_k: 50, stop_tokens: Vec::new() }
    }
}
