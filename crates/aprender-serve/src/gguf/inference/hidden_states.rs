// Final-layer hidden states for the quantized (GGUF) backend.
//
// aprender#2376 finding 1, seventh route: `/realize/embed` answered
// "No model available" on every `apr serve run model.gguf`, because the only
// hidden-state accessor in the crate was `layers::Model::forward_hidden` — a
// method on the DENSE f32 transformer, which is `None` whenever the weights
// arrived as a quantized GGUF. This is the quantized equivalent, so the
// embedding handlers have a backend on the path every `apr serve run` user hits.

impl OwnedQuantizedModel {
    /// Final-layer hidden states (post output-norm, pre `lm_head`) for each token.
    ///
    /// Returns `token_ids.len() * hidden_dim` f32s, row-major: token `t` occupies
    /// `[t * hidden_dim .. (t + 1) * hidden_dim]`. This is exactly the tensor the
    /// `lm_head` projection consumes, i.e. the same quantity
    /// [`crate::layers::Model::forward_hidden`] returns for the dense backend, so
    /// the two produce comparable sentence embeddings after pooling.
    ///
    /// Tokens are run through the production
    /// [`Self::forward_single_with_scratch`] path with a KV cache, so position `t`
    /// attends over `0..=t` exactly as it does during generation — the vectors are
    /// contextual, not per-token lookups.
    ///
    /// # Errors
    ///
    /// - [`RealizarError::InvalidShape`] if `token_ids` is empty.
    /// - [`RealizarError::ContextLimitExceeded`] if the sequence is longer than the
    ///   model's context window. Sizing the KV cache from a caller-supplied length
    ///   without this check is what let one HTTP request abort the process
    ///   (aprender#2376 finding 9).
    /// - [`RealizarError::UnsupportedOperation`] for encoder-decoder models
    ///   (T5/Whisper), whose forward pass is a different orchestrator.
    pub fn forward_hidden_states(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }
        if token_ids.len() > self.config.context_length {
            return Err(RealizarError::ContextLimitExceeded {
                provided: token_ids.len(),
                maximum: self.config.context_length,
            });
        }
        if !self.encoder_layers.is_empty() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "forward_hidden_states".to_string(),
                reason: "encoder-decoder models (T5/Whisper) are not supported".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let mut scratch = InferenceScratchBuffer::from_config(&self.config);
        let mut cache = OwnedQuantizedKVCache::from_config(&self.config, token_ids.len());

        let mut out = Vec::with_capacity(token_ids.len() * hidden_dim);
        for (position, &token_id) in token_ids.iter().enumerate() {
            self.forward_single_with_scratch(token_id, &mut cache, position, &mut scratch)?;
            // `forward_single_with_scratch` leaves the final-norm output in
            // `scratch.normed` — it is what step 4 feeds to `lm_head`.
            out.extend_from_slice(&scratch.normed[..hidden_dim]);
        }
        Ok(out)
    }
}
