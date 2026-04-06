
impl OwnedQuantizedModel {

    /// FlashAttention: Tiled attention with O(N) memory (PARITY-026)
    ///
    /// Implements the FlashAttention algorithm from Dao et al. for memory-efficient attention.
    /// Uses online softmax to process attention in tiles without materializing the N×N matrix.
    ///
    /// # Key Properties
    /// - Memory: O(N) instead of O(N²)
    /// - Numerically equivalent to standard attention
    /// - 10-100x memory savings for long sequences
    ///
    /// # Arguments
    /// * `q` - Query vector [hidden_dim]
    /// * `k_cache` - Cached keys [cache_len, hidden_dim]
    /// * `v_cache` - Cached values [cache_len, hidden_dim]
    /// * `current_k` - Current key [hidden_dim]
    /// * `current_v` - Current value [hidden_dim]
    /// * `block_size` - Tile size for tiled computation (default: 64)
    ///
    /// # Returns
    /// Attention output [hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn flash_attention_tiled(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        block_size: usize,
    ) -> Vec<f32> {
        contract_pre_flash_attention!();
        let num_heads = self.config.num_heads;
        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let head_dim = self.config.head_dim();
        let q_dim = self.config.q_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Total sequence length = cached + 1 (current)
        let cache_len = k_cache.len() / q_dim;
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; q_dim];

        // Process each head with FlashAttention tiling
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let q_head = &q[head_offset..head_offset + head_dim];

            // Online softmax state for this head
            let mut m_i = f32::NEG_INFINITY; // Running max
            let mut l_i = 0.0f32; // Running sum of exp(score - max)
            let mut o_i = vec![0.0f32; head_dim]; // Accumulated output

            // Process KV cache in tiles
            let num_tiles = total_len.div_ceil(block_size);

            for tile_idx in 0..num_tiles {
                let tile_start = tile_idx * block_size;
                let tile_end = (tile_start + block_size).min(total_len);
                let tile_len = tile_end - tile_start;

                // Compute scores for this tile
                let mut tile_scores = Vec::with_capacity(tile_len);
                let mut tile_values: Vec<&[f32]> = Vec::with_capacity(tile_len);

                for pos in tile_start..tile_end {
                    if pos < cache_len {
                        // From cache
                        let k_start = pos * q_dim + head_offset;
                        let cached_key = &k_cache[k_start..k_start + head_dim];

                        // Compute Q·K score
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_head[d] * cached_key[d];
                        }
                        tile_scores.push(score * scale);

                        let v_start = pos * q_dim + head_offset;
                        tile_values.push(&v_cache[v_start..v_start + head_dim]);
                    } else {
                        // Current position
                        let curr_key = &current_k[head_offset..head_offset + head_dim];

                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_head[d] * curr_key[d];
                        }
                        tile_scores.push(score * scale);

                        tile_values.push(&current_v[head_offset..head_offset + head_dim]);
                    }
                }

                // Find max in this tile
                let m_tile = tile_scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // Update running max
                let m_new = m_i.max(m_tile);

                // Rescale factors for online softmax
                let scale_old = (m_i - m_new).exp();
                let scale_tile = (m_tile - m_new).exp();

                // Compute local softmax sum for this tile
                let l_tile: f32 = tile_scores.iter().map(|&s| (s - m_tile).exp()).sum();

                // Update running sum
                l_i = l_i * scale_old + l_tile * scale_tile;

                // Update output: rescale old output and add new contribution
                for o in &mut o_i {
                    *o *= scale_old;
                }

                // Add weighted values from this tile
                for (j, &score) in tile_scores.iter().enumerate() {
                    let attn_weight = (score - m_tile).exp() * scale_tile;
                    let v = tile_values[j];
                    for d in 0..head_dim {
                        o_i[d] += attn_weight * v[d];
                    }
                }

                m_i = m_new;
            }

            // Finalize: divide by sum
            if l_i > 0.0 {
                for d in 0..head_dim {
                    output[head_offset + d] = o_i[d] / l_i;
                }
            }
        }

        contract_post_flash_attention!(&output);
        output
    }

    /// CPU fallback for flash_attention_tiled - uses standard attention
    #[cfg(not(feature = "gpu"))]
    #[allow(unused_variables)]
    pub fn flash_attention_tiled(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        block_size: usize,
    ) -> Vec<f32> {
        // FlashAttention is a GPU optimization; CPU path uses standard attention
        self.attention_with_cache(q, k_cache, v_cache, current_k, current_v)
    }
}

include!("rope.rs");
include!("cache.rs");
