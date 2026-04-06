impl OwnedQuantizedModelCuda {
    /// PAR-103: Pre-cache all weights for batched forward pass.
    ///
    /// This loads all layer weights into GPU memory with the naming convention
    /// expected by `forward_batch_cuda_native`. Required before using batch mode.
    ///
    /// # Returns
    ///
    /// Total bytes of weights uploaded to GPU.
    ///
    /// # Errors
    ///
    /// Returns error if weight upload fails.
    pub fn pre_cache_weights_for_batch(&mut self) -> Result<usize> {
        let mut total_bytes = 0usize;
        let num_layers = self.model.layers.len();

        eprintln!(
            "[PAR-103] Pre-caching {} layer weights for batch mode...",
            num_layers
        );

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("layer.{}", layer_idx);

            // Cache QKV weights
            match &layer.qkv_weight {
                OwnedQKVWeights::Separate { q, k, v } => {
                    let q_name = format!("{}.attn_q.weight", prefix);
                    let k_name = format!("{}.attn_k.weight", prefix);
                    let v_name = format!("{}.attn_v.weight", prefix);

                    total_bytes += self
                        .executor
                        .load_quantized_weights(&q_name, &q.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache Q weights: {}", e),
                        })?;
                    total_bytes += self
                        .executor
                        .load_quantized_weights(&k_name, &k.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache K weights: {}", e),
                        })?;
                    total_bytes += self
                        .executor
                        .load_quantized_weights(&v_name, &v.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache V weights: {}", e),
                        })?;
                },
                OwnedQKVWeights::Fused(qkv) => {
                    let qkv_name = format!("{}.attn_qkv.weight", prefix);
                    total_bytes += self
                        .executor
                        .load_quantized_weights(&qkv_name, &qkv.data)
                        .map_err(|e| RealizarError::UnsupportedOperation {
                            operation: "pre_cache_weights_for_batch".to_string(),
                            reason: format!("Failed to cache QKV weights: {}", e),
                        })?;
                },
            }

            // Cache O projection
            let o_name = format!("{}.attn_output.weight", prefix);
            total_bytes += self
                .executor
                .load_quantized_weights(&o_name, &layer.attn_output_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "pre_cache_weights_for_batch".to_string(),
                    reason: format!("Failed to cache O weights: {}", e),
                })?;

            // Cache FFN weights (ffn_gate is optional - only SwiGLU models have it)
            let ffn_up_name = format!("{}.ffn_up.weight", prefix);
            let ffn_down_name = format!("{}.ffn_down.weight", prefix);

            total_bytes += self
                .executor
                .load_quantized_weights(&ffn_up_name, &layer.ffn_up_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "pre_cache_weights_for_batch".to_string(),
                    reason: format!("Failed to cache FFN up weights: {}", e),
                })?;
            total_bytes += self
                .executor
                .load_quantized_weights(&ffn_down_name, &layer.ffn_down_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "pre_cache_weights_for_batch".to_string(),
                    reason: format!("Failed to cache FFN down weights: {}", e),
                })?;

            // FFN gate weight present for gated FFN architectures (SwiGLU/GatedMLP)
            if let Some(ref gate_weight) = layer.ffn_gate_weight {
                let ffn_gate_name = format!("{}.ffn_gate.weight", prefix);
                total_bytes += self
                    .executor
                    .load_quantized_weights(&ffn_gate_name, &gate_weight.data)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "pre_cache_weights_for_batch".to_string(),
                        reason: format!("Failed to cache FFN gate weights: {}", e),
                    })?;
            }
        }

        // Cache LM head
        let lm_head_name = "output.weight".to_string();
        total_bytes += self
            .executor
            .load_quantized_weights(&lm_head_name, &self.model.lm_head_weight.data)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "pre_cache_weights_for_batch".to_string(),
                reason: format!("Failed to cache LM head weights: {}", e),
            })?;

        let total_mb = total_bytes / (1024 * 1024);
        eprintln!(
            "[PAR-103] Pre-cached {} MB of weights for batch mode",
            total_mb
        );
        Ok(total_bytes)
    }
}
