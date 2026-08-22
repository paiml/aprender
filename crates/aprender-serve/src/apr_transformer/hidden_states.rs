// Final-layer hidden states for the `AprTransformer` (f32 APR / SafeTensors) backend.
//
// aprender#2609: the crate had exactly two hidden-state accessors —
// `layers::Model::forward_hidden` (dense f32 `Model`) and
// `gguf::OwnedQuantizedModel::forward_hidden_states` (quantized, added by
// aprender#2376). The third resident backend, `AppState::apr_transformer`, had
// neither, so on a server holding one the embedding routes fell through every
// arm of `resolve_embed_backend` and answered "No model available" while
// `/health` reported `model_loaded: true` and `/generate` returned tokens.

impl AprTransformer {
    /// Final-layer hidden states (post output-norm, pre `lm_head`) for each token.
    ///
    /// Returns `token_ids.len() * hidden_dim` f32s, row-major: token `t` occupies
    /// `[t * hidden_dim .. (t + 1) * hidden_dim]`.
    ///
    /// Tokens are run through the production
    /// [`Self::forward_hidden_with_cache`] path with a KV cache, so position `t`
    /// attends over `0..=t` exactly as it does during generation — the vectors are
    /// contextual, not per-token embedding lookups.
    ///
    /// # Errors
    ///
    /// - [`RealizarError::InvalidShape`] if `token_ids` is empty.
    /// - [`RealizarError::ContextLimitExceeded`] if the sequence is longer than the
    ///   model's context window. Sizing a cache from a caller-supplied length with
    ///   no ceiling is what let one HTTP request abort the process
    ///   (aprender#2376 finding 9), so the check is here and not at the caller.
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

        let hidden_dim = self.config.hidden_dim;
        let mut cache = AprKVCache::new(&self.config);
        let mut out = Vec::with_capacity(token_ids.len() * hidden_dim);
        for (position, &token_id) in token_ids.iter().enumerate() {
            let normed = self.forward_hidden_with_cache(token_id, &mut cache, position)?;
            out.extend_from_slice(&normed[..hidden_dim]);
        }
        Ok(out)
    }
}
