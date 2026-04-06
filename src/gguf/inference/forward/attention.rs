impl OwnedQuantizedModel {

    // =========================================================================
    // IMP-110: Multi-Head Parallel Attention
    // =========================================================================

    /// Reshape tensor from [seq_len, hidden_dim] to [num_heads, seq_len, head_dim]
    ///
    /// IMP-110b: Prepares Q/K/V tensors for parallel multi-head processing.
    /// Original layout stores all head features contiguously per position.
    /// New layout groups by head for batched matmul operations.
    ///
    /// # Arguments
    /// * `input` - Input tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head (hidden_dim / num_heads)
    ///
    /// # Returns
    /// Reshaped tensor [num_heads, seq_len, head_dim]
    #[cfg(feature = "gpu")]
    pub fn reshape_for_parallel_heads(
        &self,
        input: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = num_heads * head_dim;
        let expected_len = seq_len * hidden_dim;

        if input.len() != expected_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match seq_len={} * hidden_dim={}={}",
                    input.len(),
                    seq_len,
                    hidden_dim,
                    expected_len
                ),
            });
        }

        let mut reshaped = vec![0.0f32; num_heads * seq_len * head_dim];

        // Transform: input[pos * hidden_dim + h * head_dim + d]
        //         -> reshaped[h * seq_len * head_dim + pos * head_dim + d]
        for h in 0..num_heads {
            for pos in 0..seq_len {
                for d in 0..head_dim {
                    let orig_idx = pos * hidden_dim + h * head_dim + d;
                    let new_idx = h * seq_len * head_dim + pos * head_dim + d;
                    reshaped[new_idx] = input[orig_idx];
                }
            }
        }

        Ok(reshaped)
    }

    /// Compute batched Q@K^T scores for all heads in parallel
    ///
    /// IMP-110c: Computes attention scores for all heads in a single batch.
    /// Takes Q, K in original [seq_len, hidden_dim] layout and computes
    /// Q@K^T for each head.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `scale` - Attention scale (1/sqrt(head_dim))
    ///
    /// # Returns
    /// Batched scores [num_heads, seq_len, seq_len]
    #[cfg(feature = "gpu")]
    pub fn parallel_batched_qk_scores(
        &self,
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        // Reshape Q and K to [num_heads, seq_len, head_dim]
        let q_reshaped = self.reshape_for_parallel_heads(q, seq_len, num_heads, head_dim)?;
        let k_reshaped = self.reshape_for_parallel_heads(k, seq_len, num_heads, head_dim)?;

        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::with_threshold".to_string(),
                reason: format!("GPU scheduler initialization failed: {e}"),
            }
        })?;

        // For each head: Q_h @ K_h^T -> [seq_len, seq_len]
        // Total output: [num_heads, seq_len, seq_len]
        let mut all_scores = Vec::with_capacity(num_heads * seq_len * seq_len);

        for h in 0..num_heads {
            let head_start = h * seq_len * head_dim;
            let q_h = &q_reshaped[head_start..head_start + seq_len * head_dim];
            let k_h = &k_reshaped[head_start..head_start + seq_len * head_dim];

            // Transpose K_h: [seq_len, head_dim] -> [head_dim, seq_len]
            let mut k_t = vec![0.0f32; head_dim * seq_len];
            for i in 0..seq_len {
                for j in 0..head_dim {
                    k_t[j * seq_len + i] = k_h[i * head_dim + j];
                }
            }

            // Q_h @ K_h^T: [seq_len, head_dim] @ [head_dim, seq_len] -> [seq_len, seq_len]
            let scores = scheduler
                .matmul(q_h, &k_t, seq_len, head_dim, seq_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "parallel_batched_qk_scores".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Apply scale and accumulate
            for s in &scores {
                all_scores.push(s * scale);
            }
        }

        Ok(all_scores)
    }

    /// Multi-head attention with parallel head processing
    ///
    /// IMP-110a: Processes all attention heads in parallel batches instead
    /// of iterating head-by-head. This enables better GPU utilization.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `v` - Value tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Attention output [seq_len, hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn parallel_multihead_attention_gpu(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Get batched scores for all heads: [num_heads, seq_len, seq_len]
        let batched_scores =
            self.parallel_batched_qk_scores(q, k, seq_len, num_heads, head_dim, scale)?;

        // Apply causal mask and softmax per head
        let mut batched_weights = vec![0.0f32; num_heads * seq_len * seq_len];
        for h in 0..num_heads {
            let head_offset = h * seq_len * seq_len;
            let head_scores = &batched_scores[head_offset..head_offset + seq_len * seq_len];
            let head_weights = self.apply_causal_mask_softmax(head_scores, seq_len);
            batched_weights[head_offset..head_offset + seq_len * seq_len]
                .copy_from_slice(&head_weights);
        }

        // Reshape V to [num_heads, seq_len, head_dim]
        let v_reshaped = self.reshape_for_parallel_heads(v, seq_len, num_heads, head_dim)?;

        // Compute attention output for all heads
        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "HybridScheduler::with_threshold".to_string(),
                reason: format!("GPU scheduler initialization failed: {e}"),
            }
        })?;

        // Output: [seq_len, hidden_dim]
        let mut output = vec![0.0f32; seq_len * hidden_dim];

        for h in 0..num_heads {
            let weights_offset = h * seq_len * seq_len;
            let v_offset = h * seq_len * head_dim;

            let head_weights = &batched_weights[weights_offset..weights_offset + seq_len * seq_len];
            let v_h = &v_reshaped[v_offset..v_offset + seq_len * head_dim];

            // weights @ V_h: [seq_len, seq_len] @ [seq_len, head_dim] -> [seq_len, head_dim]
            let head_output = scheduler
                .matmul(head_weights, v_h, seq_len, seq_len, head_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "parallel_multihead_attention_gpu".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Copy to output in original layout
            for pos in 0..seq_len {
                let out_start = pos * hidden_dim + h * head_dim;
                let head_start = pos * head_dim;
                output[out_start..out_start + head_dim]
                    .copy_from_slice(&head_output[head_start..head_start + head_dim]);
            }
        }

        Ok(output)
    }

    // =========================================================================
    // IMP-111: Flash Attention-style Tiled Computation
    // =========================================================================

    /// Standard softmax (reference implementation)
    ///
    /// IMP-111a: Reference implementation for testing online softmax.
    /// Computes softmax in the standard way: exp(x - max) / sum(exp(x - max))
    pub fn standard_softmax(&self, scores: &[f32]) -> Vec<f32> {
        contract_pre_standard_softmax!();
        if scores.is_empty() {
            return Vec::new();
        }

        // Find max for numerical stability
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) and sum
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();

        // Normalize
        let result: Vec<f32> = exp_scores.iter().map(|&e| e / sum).collect();
        contract_post_softmax!(&result);
        result
    }

    /// Online softmax with tiled processing (O(1) memory per tile)
    ///
    /// IMP-111a: Implements the "online softmax" algorithm that processes
    /// data in tiles without materializing the full softmax denominator.
    ///
    /// Algorithm:
    /// 1. Process tiles, tracking running max (m) and denominator (d)
    /// 2. When new tile has larger max, rescale previous denominator
    /// 3. Final pass normalizes all values
    ///
    /// # Arguments
    /// * `scores` - Input scores to apply softmax
    /// * `tile_size` - Size of each tile for processing
    ///
    /// # Returns
    /// Softmax probabilities
    pub fn online_softmax(&self, scores: &[f32], tile_size: usize) -> Result<Vec<f32>> {
        if scores.is_empty() {
            return Ok(Vec::new());
        }

        let n = scores.len();
        let tile_size = tile_size.max(1);

        // Running statistics
        let mut global_max = f32::NEG_INFINITY;
        let mut global_sum = 0.0f32;

        // First pass: compute global max and sum using online algorithm
        for tile_start in (0..n).step_by(tile_size) {
            let tile_end = (tile_start + tile_size).min(n);

            // Find local max in this tile
            let local_max = scores[tile_start..tile_end]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            if local_max > global_max {
                // Rescale previous sum when we find a new max
                let rescale = (global_max - local_max).exp();
                global_sum *= rescale;
                global_max = local_max;
            }

            // Add this tile's contribution to sum
            for &s in &scores[tile_start..tile_end] {
                global_sum += (s - global_max).exp();
            }
        }

        // Second pass: compute final softmax values
        let mut result = Vec::with_capacity(n);
        for &s in scores {
            result.push((s - global_max).exp() / global_sum);
        }

        Ok(result)
    }

    /// Standard single-head attention (reference implementation)
    ///
    /// IMP-111b: Reference implementation that materializes full attention matrix.
    /// Used to verify tiled attention correctness.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Dimension per head
    /// * `scale` - Attention scale (1/sqrt(head_dim))
    pub fn standard_single_head_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Compute attention scores: Q @ K^T -> [seq_len, seq_len]
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }

        // Apply softmax per row
        let mut weights = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row = &scores[row_start..row_start + seq_len];
            let softmax = self.standard_softmax(row);
            weights[row_start..row_start + seq_len].copy_from_slice(&softmax);
        }

        // Compute output: weights @ V -> [seq_len, head_dim]
        let mut output = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += weights[i * seq_len + j] * v[j * head_dim + d];
                }
                output[i * head_dim + d] = acc;
            }
        }

        Ok(output)
    }

    /// Tiled single-head attention (non-causal)
    ///
    /// IMP-111b: Flash Attention-style tiled computation.
    /// Processes K/V in tiles, maintaining running softmax statistics.
    #[allow(clippy::too_many_arguments)]
    pub fn tiled_single_head_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        tile_size: usize,
    ) -> Result<Vec<f32>> {
        let tile_size = tile_size.max(1);
        let mut output = vec![0.0f32; seq_len * head_dim];

        // Process each query position
        for i in 0..seq_len {
            let q_i = &q[i * head_dim..(i + 1) * head_dim];

            // Running statistics for online softmax
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            let mut running_output = vec![0.0f32; head_dim];

            // Process K/V in tiles
            for tile_start in (0..seq_len).step_by(tile_size) {
                let tile_end = (tile_start + tile_size).min(seq_len);

                // Compute scores for this tile: q_i @ K_tile^T
                let mut tile_scores = Vec::with_capacity(tile_end - tile_start);
                for j in tile_start..tile_end {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_i[d] * k[j * head_dim + d];
                    }
                    tile_scores.push(dot * scale);
                }

                // Find tile max
                let tile_max = tile_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Update running statistics
                let new_max = running_max.max(tile_max);

                // Rescale previous output and sum
                if new_max > running_max && running_sum > 0.0 {
                    let rescale = (running_max - new_max).exp();
                    running_sum *= rescale;
                    for out_val in &mut running_output {
                        *out_val *= rescale;
                    }
                }
                running_max = new_max;

                // Accumulate this tile's contribution
                for (idx, &score) in tile_scores.iter().enumerate() {
                    let j = tile_start + idx;
                    let weight = (score - running_max).exp();
                    running_sum += weight;
                    for d in 0..head_dim {
                        running_output[d] += weight * v[j * head_dim + d];
                    }
                }
            }

            // Normalize output
            for d in 0..head_dim {
                output[i * head_dim + d] = running_output[d] / running_sum;
            }
        }

        Ok(output)
    }
}
