impl AprV2ModelCuda {

    // ========================================================================
    // GPU-accelerated inference
    // ========================================================================

    /// GPU-accelerated forward pass returning only the next token ID (fastest path).
    ///
    /// Uses GPU argmax to avoid transferring 600KB of logits from GPU to CPU.
    /// This is the recommended method for autoregressive generation.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Input token ID (single token for decode step)
    ///
    /// # Returns
    ///
    /// The token ID with the highest logit value.
    pub fn forward_cuda_to_token(&mut self, token_id: u32) -> Result<u32> {
        // GH-282: Ensure CUDA context is current for this thread
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        if !self.model.metadata.is_transformer() {
            return Err(RealizarError::FormatError {
                reason: "Model is not a transformer (missing config)".to_string(),
            });
        }

        let _hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let _num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let vocab_size = self.model.metadata.vocab_size.unwrap_or(0);

        // Use indexed Q4K path with GPU argmax (no 600KB logits transfer)
        // Phase 45: Skip fast path when test_executor is present
        // GH-201: Skip fast path in streaming mode (layer weights not pre-cached)
        if self.test_executor.is_none()
            && self.executor.has_indexed_weights()
            && !self.streaming_mode
        {
            let position = self.kv_position;

            // Embedding lookup from cache
            let input: Vec<f32> = self
                .get_embedding(token_id)
                .ok_or_else(|| RealizarError::InvalidShape {
                    reason: format!("Token {} out of embedding range", token_id),
                })?
                .to_vec();

            let num_layers = self.model.metadata.num_layers.unwrap_or(0);
            let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
            let intermediate_dim = self
                .model
                .metadata
                .intermediate_size
                .unwrap_or(hidden_dim * 4);
            let eps = self.model.metadata.rms_norm_eps.unwrap_or(1e-6);

            // First call: capture graph using the full graphed forward path
            // Subsequent calls: use replay with GPU argmax
            let next_token = if !self.executor.has_decode_graph() {
                // Need to capture graph first - use forward_all_layers_gpu_to_logits_graphed
                // then do CPU argmax
                let mut output = vec![0.0f32; vocab_size];
                self.executor
                    .forward_all_layers_gpu_to_logits_graphed(
                        &input,
                        &mut output,
                        position,
                        num_layers,
                        hidden_dim as u32,
                        intermediate_dim as u32,
                        vocab_size as u32,
                        eps,
                    )
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "forward_all_layers_gpu_to_logits_graphed".to_string(),
                        reason: format!("Graph capture failed: {e}"),
                    })?;

                // CPU argmax for first token (graph now captured)
                let (top_idx, _) = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .ok_or_else(|| RealizarError::InvalidShape {
                        reason: "Empty logits".to_string(),
                    })?;
                top_idx as u32
            } else {
                // Graph captured - use fast replay with GPU argmax
                self.executor
                    .forward_graphed_replay_to_token_id(&input, position, vocab_size as u32)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "forward_graphed_replay_to_token_id".to_string(),
                        reason: format!("GPU argmax fast path failed: {e}"),
                    })?
            };

            // Increment position for next token
            self.kv_position += 1;

            return Ok(next_token);
        }

        // Fallback: use forward_cuda and do CPU argmax
        let logits = self.forward_cuda(&[token_id])?;
        let (top_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;
        Ok(top_idx as u32)
    }
}
