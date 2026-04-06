impl CudaExecutor {

    // ========================================================================
    // QWEN-007: Q8 Quantized KV Cache for 4x Memory Reduction
    // ========================================================================

    /// Initialize Q8 quantized KV cache for memory-constrained inference
    ///
    /// Q8 format stores KV vectors as INT8 with per-block FP32 scales,
    /// reducing memory by 4x compared to FP32 while maintaining quality.
    ///
    /// Memory comparison for 32-layer, 8 KV-head, 128-dim, 4096 seq model:
    /// - FP32: 32 × 2 × 4096 × 8 × 128 × 4 = 8.59 GB
    /// - Q8:   32 × 2 × 4096 × 8 × 128 × 1 + scales = 2.15 GB (4x reduction)
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `head_dim` - Dimension per head (should be divisible by 32)
    /// * `max_len` - Maximum sequence length to support
    #[allow(clippy::too_many_arguments)]
    pub fn init_kv_cache_q8_gpu(
        &mut self,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_len: usize,
    ) -> Result<(), GpuError> {
        // Validate head_dim is divisible by 32 (Q8 block size)
        if !head_dim.is_multiple_of(32) {
            return Err(GpuError::InvalidParameter(format!(
                "QWEN-007: head_dim must be divisible by 32 for Q8 quantization, got {}",
                head_dim
            )));
        }

        // Store dimensions
        self.kv_num_heads = num_heads;
        self.kv_num_kv_heads = num_kv_heads;
        self.kv_head_dim = head_dim;
        self.kv_cache_max_len = max_len;
        self.kv_cache_q8_enabled = true;

        // Q8 buffer sizes
        // Values: [num_kv_heads × max_len × head_dim] as i8 (1 byte each)
        let values_size = num_kv_heads * max_len * head_dim;
        // Scales: [num_kv_heads × max_len × (head_dim / 32)] as f32 (4 bytes each)
        let scales_size = num_kv_heads * max_len * (head_dim / 32);

        for layer_idx in 0..num_layers {
            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);

            // Allocate Q8 buffers if not already present
            if !self.kv_cache_q8_k.contains_key(&k_key) {
                // Quantized values (i8)
                let k_buf = GpuBuffer::<i8>::new(&self.context, values_size)?;
                let v_buf = GpuBuffer::<i8>::new(&self.context, values_size)?;
                self.kv_cache_q8_k.insert(k_key.clone(), k_buf);
                self.kv_cache_q8_v.insert(v_key.clone(), v_buf);

                // Scales (f32)
                let k_scales_key = format!("kv_{}_k_scales", layer_idx);
                let v_scales_key = format!("kv_{}_v_scales", layer_idx);
                let k_scales = GpuBuffer::<f32>::new(&self.context, scales_size)?;
                let v_scales = GpuBuffer::<f32>::new(&self.context, scales_size)?;
                self.kv_cache_q8_k_scales.insert(k_scales_key, k_scales);
                self.kv_cache_q8_v_scales.insert(v_scales_key, v_scales);

                self.kv_cache_lengths.insert(layer_idx, 0);
            }
        }

        // Total memory: values (1 byte) + scales (4 bytes per 32 values)
        let total_bytes = num_layers * 2 * (values_size + scales_size * 4);
        self.memory_pool.record_allocation(total_bytes);

        Ok(())
    }

    /// Check if Q8 KV cache mode is enabled
    #[must_use]
    pub fn is_kv_cache_q8_enabled(&self) -> bool {
        self.kv_cache_q8_enabled
    }

    /// Get Q8 KV cache memory usage in bytes
    #[must_use]
    pub fn kv_cache_q8_memory_bytes(&self) -> usize {
        if !self.kv_cache_q8_enabled {
            return 0;
        }
        let num_layers = self.kv_cache_lengths.len();
        let values_size = self.kv_num_kv_heads * self.kv_cache_max_len * self.kv_head_dim;
        let scales_size = self.kv_num_kv_heads * self.kv_cache_max_len * (self.kv_head_dim / 32);
        num_layers * 2 * (values_size + scales_size * 4)
    }

    /// Get FP32 equivalent memory for comparison (what would be used without Q8)
    #[must_use]
    pub fn kv_cache_fp32_equivalent_bytes(&self) -> usize {
        let num_layers = self.kv_cache_lengths.len();
        let fp32_size = self.kv_num_kv_heads * self.kv_cache_max_len * self.kv_head_dim * 4;
        num_layers * 2 * fp32_size
    }

    /// Write K/V vectors to Q8 cache at a specific position
    ///
    /// Quantizes FP32 K/V vectors to Q8 format and uploads to GPU.
    /// This is the Phase 2 implementation using CPU quantization.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index
    /// * `position` - Position in the sequence (0-indexed)
    /// * `k` - Key vector [num_kv_heads × head_dim]
    /// * `v` - Value vector [num_kv_heads × head_dim]
    pub fn write_kv_q8(
        &mut self,
        layer_idx: usize,
        position: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), GpuError> {
        if !self.kv_cache_q8_enabled {
            return Err(GpuError::InvalidParameter(
                "Q8 KV cache not enabled. Call init_kv_cache_q8_gpu first.".to_string(),
            ));
        }

        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;

        // Validate input sizes
        let expected_size = num_kv_heads * head_dim;
        if k.len() != expected_size || v.len() != expected_size {
            return Err(GpuError::InvalidParameter(format!(
                "K/V size mismatch: expected {}, got K={}, V={}",
                expected_size,
                k.len(),
                v.len()
            )));
        }

        if position >= max_len {
            return Err(GpuError::InvalidParameter(format!(
                "Position {} exceeds max_len {}",
                position, max_len
            )));
        }

        // Quantize K and V to Q8 (CPU-side for Phase 2)
        let blocks_per_head = head_dim / 32;
        let mut k_quants = Vec::with_capacity(expected_size);
        let mut k_scales = Vec::with_capacity(num_kv_heads * blocks_per_head);
        let mut v_quants = Vec::with_capacity(expected_size);
        let mut v_scales = Vec::with_capacity(num_kv_heads * blocks_per_head);

        for head in 0..num_kv_heads {
            for block_idx in 0..blocks_per_head {
                let start = head * head_dim + block_idx * 32;
                let k_block_vals: [f32; 32] = k[start..start + 32].try_into().map_err(|_| {
                    GpuError::InvalidParameter("K block extraction failed".to_string())
                })?;
                let v_block_vals: [f32; 32] = v[start..start + 32].try_into().map_err(|_| {
                    GpuError::InvalidParameter("V block extraction failed".to_string())
                })?;

                let k_block = Q8_0Block::quantize(&k_block_vals);
                let v_block = Q8_0Block::quantize(&v_block_vals);

                k_quants.extend_from_slice(&k_block.quants);
                k_scales.push(k_block.scale);
                v_quants.extend_from_slice(&v_block.quants);
                v_scales.push(v_block.scale);
            }
        }

        // Upload to GPU at the correct position offset
        // Layout: [num_kv_heads, max_len, head_dim]
        // Position offset = position * head_dim for each head
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);
        let k_scales_key = format!("kv_{}_k_scales", layer_idx);
        let v_scales_key = format!("kv_{}_v_scales", layer_idx);

        // Get mutable references to buffers
        let k_buf = self.kv_cache_q8_k.get_mut(&k_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Q8 K cache for layer {} not found", layer_idx))
        })?;
        let k_scales_buf = self
            .kv_cache_q8_k_scales
            .get_mut(&k_scales_key)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "Q8 K scales for layer {} not found",
                    layer_idx
                ))
            })?;

        // Upload K quants and scales for each head at the correct position
        for head in 0..num_kv_heads {
            let src_q_offset = head * head_dim;
            let dst_q_offset = head * max_len * head_dim + position * head_dim;
            k_buf.copy_from_host_at(
                &k_quants[src_q_offset..src_q_offset + head_dim],
                dst_q_offset,
            )?;

            let src_s_offset = head * blocks_per_head;
            let dst_s_offset = head * max_len * blocks_per_head + position * blocks_per_head;
            k_scales_buf.copy_from_host_at(
                &k_scales[src_s_offset..src_s_offset + blocks_per_head],
                dst_s_offset,
            )?;
        }

        // Same for V
        let v_buf = self.kv_cache_q8_v.get_mut(&v_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Q8 V cache for layer {} not found", layer_idx))
        })?;
        let v_scales_buf = self
            .kv_cache_q8_v_scales
            .get_mut(&v_scales_key)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "Q8 V scales for layer {} not found",
                    layer_idx
                ))
            })?;

        for head in 0..num_kv_heads {
            let src_q_offset = head * head_dim;
            let dst_q_offset = head * max_len * head_dim + position * head_dim;
            v_buf.copy_from_host_at(
                &v_quants[src_q_offset..src_q_offset + head_dim],
                dst_q_offset,
            )?;

            let src_s_offset = head * blocks_per_head;
            let dst_s_offset = head * max_len * blocks_per_head + position * blocks_per_head;
            v_scales_buf.copy_from_host_at(
                &v_scales[src_s_offset..src_s_offset + blocks_per_head],
                dst_s_offset,
            )?;
        }

        // Update cache length
        let current_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        if position >= current_len {
            self.kv_cache_lengths.insert(layer_idx, position + 1);
        }

        Ok(())
    }

    /// Read K/V vectors from Q8 cache at a range of positions
    ///
    /// Downloads Q8 data from GPU and dequantizes to FP32.
    /// This is the Phase 2 implementation using CPU dequantization.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Layer index
    /// * `start_pos` - Start position (inclusive)
    /// * `end_pos` - End position (exclusive)
    ///
    /// # Returns
    ///
    /// Tuple of (K, V) vectors, each [seq_len × num_kv_heads × head_dim]
    pub fn read_kv_q8(
        &self,
        layer_idx: usize,
        start_pos: usize,
        end_pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), GpuError> {
        if !self.kv_cache_q8_enabled {
            return Err(GpuError::InvalidParameter(
                "Q8 KV cache not enabled. Call init_kv_cache_q8_gpu first.".to_string(),
            ));
        }

        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let blocks_per_head = head_dim / 32;
        let seq_len = end_pos.saturating_sub(start_pos);

        if end_pos > max_len {
            return Err(GpuError::InvalidParameter(format!(
                "End position {} exceeds max_len {}",
                end_pos, max_len
            )));
        }

        // Get buffer references
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);
        let k_scales_key = format!("kv_{}_k_scales", layer_idx);
        let v_scales_key = format!("kv_{}_v_scales", layer_idx);

        let k_buf = self.kv_cache_q8_k.get(&k_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Q8 K cache for layer {} not found", layer_idx))
        })?;
        let k_scales_buf = self
            .kv_cache_q8_k_scales
            .get(&k_scales_key)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "Q8 K scales for layer {} not found",
                    layer_idx
                ))
            })?;
        let v_buf = self.kv_cache_q8_v.get(&v_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Q8 V cache for layer {} not found", layer_idx))
        })?;
        let v_scales_buf = self
            .kv_cache_q8_v_scales
            .get(&v_scales_key)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "Q8 V scales for layer {} not found",
                    layer_idx
                ))
            })?;

        // Download quantized data for each position
        let mut k_out = Vec::with_capacity(seq_len * num_kv_heads * head_dim);
        let mut v_out = Vec::with_capacity(seq_len * num_kv_heads * head_dim);

        for pos in start_pos..end_pos {
            for head in 0..num_kv_heads {
                // Download K quants and scales
                let q_offset = head * max_len * head_dim + pos * head_dim;
                let s_offset = head * max_len * blocks_per_head + pos * blocks_per_head;

                let mut k_quants = vec![0i8; head_dim];
                let mut k_scales = vec![0.0f32; blocks_per_head];
                k_buf.copy_to_host_at(&mut k_quants, q_offset)?;
                k_scales_buf.copy_to_host_at(&mut k_scales, s_offset)?;

                let mut v_quants = vec![0i8; head_dim];
                let mut v_scales = vec![0.0f32; blocks_per_head];
                v_buf.copy_to_host_at(&mut v_quants, q_offset)?;
                v_scales_buf.copy_to_host_at(&mut v_scales, s_offset)?;

                // Dequantize
                for block_idx in 0..blocks_per_head {
                    let block_start = block_idx * 32;
                    let block = Q8_0Block {
                        scale: k_scales[block_idx],
                        quants: k_quants[block_start..block_start + 32]
                            .try_into()
                            .expect("32 values"),
                    };
                    k_out.extend_from_slice(&block.dequantize());

                    let block = Q8_0Block {
                        scale: v_scales[block_idx],
                        quants: v_quants[block_start..block_start + 32]
                            .try_into()
                            .expect("32 values"),
                    };
                    v_out.extend_from_slice(&block.dequantize());
                }
            }
        }

        Ok((k_out, v_out))
    }
}
