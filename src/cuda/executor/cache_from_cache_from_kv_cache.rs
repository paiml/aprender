impl CudaExecutor {
    // ========================================================================
    // PAR-018: GPU-Resident KV Cache for Incremental Attention
    // ========================================================================

    /// Initialize GPU KV cache for a given number of layers and max sequence length
    ///
    /// Pre-allocates GPU memory for all layers to avoid allocation during inference.
    /// Call this once at model load time with the expected max sequence length.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA, <= num_heads)
    /// * `head_dim` - Dimension per head
    /// * `max_len` - Maximum sequence length to support
    #[allow(clippy::too_many_arguments)]
    pub fn init_kv_cache_gpu(
        &mut self,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_len: usize,
    ) -> Result<(), GpuError> {
        // Store dimensions (PAR-021: track both Q heads and KV heads for GQA)
        self.kv_num_heads = num_heads;
        self.kv_num_kv_heads = num_kv_heads;
        self.kv_head_dim = head_dim;
        self.kv_cache_max_len = max_len;

        // Pre-allocate K and V buffers for each layer
        // PAR-021 GQA: Layout is [num_kv_heads, max_len, head_dim]
        let buffer_size = num_kv_heads * max_len * head_dim;

        for layer_idx in 0..num_layers {
            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);

            // Allocate if not already present
            if !self.kv_cache_gpu.contains_key(&k_key) {
                let k_buf = GpuBuffer::<f32>::new(&self.context, buffer_size)?;
                let v_buf = GpuBuffer::<f32>::new(&self.context, buffer_size)?;
                self.kv_cache_gpu.insert(k_key, k_buf);
                self.kv_cache_gpu.insert(v_key, v_buf);
                self.kv_cache_lengths.insert(layer_idx, 0);
            }
        }

        let total_bytes = num_layers * 2 * buffer_size * 4;
        self.memory_pool.record_allocation(total_bytes);

        Ok(())
    }

    /// PAR-119: Initialize batched KV caches for true multi-sequence batching
    ///
    /// Allocates M separate KV caches per layer, enabling parallel attention
    /// across M sequences. This eliminates the sequential attention bottleneck
    /// identified in Five-Whys analysis.
    ///
    /// Memory layout per layer:
    /// - K cache: [M, num_kv_heads, max_len, head_dim]
    /// - V cache: same
    /// - Stride: num_kv_heads × max_len × head_dim (per sequence)
    pub fn init_batched_kv_cache_gpu(
        &mut self,
        num_layers: usize,
        batch_size: usize,
    ) -> Result<(), GpuError> {
        // PAR-129: Extended to M=32 via 4-warp kernel
        if batch_size == 0 || batch_size > 32 {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-119: batch_size must be 1-32, got {}",
                batch_size
            )));
        }

        // Must have regular KV cache initialized first (to get dimensions)
        if self.kv_cache_max_len == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-119: Must call init_kv_cache_gpu before init_batched_kv_cache_gpu".to_string(),
            ));
        }

        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;

        // Per-sequence stride
        let stride = num_kv_heads * max_len * head_dim;
        self.batched_kv_stride = stride;

        // M× larger buffer per layer
        let buffer_size = batch_size * stride;

        // PAR-119: Check if we need to reallocate (batch_size changed)
        let need_realloc = batch_size > self.batched_kv_allocated_batch;
        if need_realloc {
            // Clear existing caches - they're too small
            self.batched_kv_k_caches.clear();
            self.batched_kv_v_caches.clear();
        }

        for layer_idx in 0..num_layers {
            // Allocate if not already present or after realloc
            if !self.batched_kv_k_caches.contains_key(&layer_idx) {
                let k_buf = GpuBuffer::<f32>::new(&self.context, buffer_size)?;
                let v_buf = GpuBuffer::<f32>::new(&self.context, buffer_size)?;
                self.batched_kv_k_caches.insert(layer_idx, k_buf);
                self.batched_kv_v_caches.insert(layer_idx, v_buf);
            }
        }

        // Track allocated batch size
        self.batched_kv_allocated_batch = batch_size;

        // Initialize per-sequence lengths (all start at 0)
        self.batched_kv_lengths = vec![0; batch_size];

        // Allocate GPU pointer arrays for batched attention
        self.batched_k_ptrs = Some(GpuBuffer::new(&self.context, batch_size)?);
        self.batched_v_ptrs = Some(GpuBuffer::new(&self.context, batch_size)?);
        self.batched_seq_lens_gpu = Some(GpuBuffer::new(&self.context, batch_size)?);

        let total_bytes = num_layers * 2 * buffer_size * 4 + batch_size * 24; // caches + ptr arrays
        self.memory_pool.record_allocation(total_bytes);

        eprintln!(
            "[PAR-119] Initialized batched KV cache: {} layers × {} sequences, stride={}, total={}MB",
            num_layers,
            batch_size,
            stride,
            total_bytes / (1024 * 1024)
        );

        Ok(())
    }

    /// PAR-119: Reset batched KV caches for new generation
    pub fn reset_batched_kv_cache_gpu(&mut self) {
        for len in &mut self.batched_kv_lengths {
            *len = 0;
        }
    }

    /// PMAT-044: Copy single KV cache to batched KV cache at a specific slot.
    ///
    /// After prefill populates the single GPU KV cache (kv_L_k, kv_L_v),
    /// this copies it into the batched KV cache at the correct stride offset
    /// for the given slot. This enables batched decode after sequential prefill.
    pub fn scatter_single_kv_to_batched(
        &mut self,
        slot_idx: usize,
        seq_len: usize,
    ) -> Result<(), GpuError> {
        if seq_len == 0 {
            return Ok(());
        }

        let stride = self.batched_kv_stride;
        if stride == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "PMAT-044: batched KV cache not initialized (stride=0)".to_string(),
            ));
        }

        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let copy_elements = num_kv_heads * seq_len * head_dim;
        let slot_offset = slot_idx * stride;
        let size_bytes = copy_elements * std::mem::size_of::<f32>();
        let offset_bytes = (slot_offset * std::mem::size_of::<f32>()) as u64;

        // Collect pointer pairs first to avoid borrow conflicts
        let layer_indices: Vec<usize> = self.batched_kv_k_caches.keys().copied().collect();
        let mut copies: Vec<(u64, u64, u64, u64)> = Vec::new(); // (batched_k_dst, single_k_src, batched_v_dst, single_v_src)

        for &layer_idx in &layer_indices {
            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);

            let single_k_ptr = self
                .kv_cache_gpu
                .get(&k_key)
                .ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-044: single KV cache '{}' not found", k_key
                    ))
                })?
                .as_ptr();
            let batched_k_ptr = self
                .batched_kv_k_caches
                .get(&layer_idx)
                .ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-044: batched K cache layer {} not found", layer_idx
                    ))
                })?
                .as_ptr();

            let single_v_ptr = self
                .kv_cache_gpu
                .get(&v_key)
                .ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-044: single KV cache '{}' not found", v_key
                    ))
                })?
                .as_ptr();
            let batched_v_ptr = self
                .batched_kv_v_caches
                .get(&layer_idx)
                .ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-044: batched V cache layer {} not found", layer_idx
                    ))
                })?
                .as_ptr();

            copies.push((
                batched_k_ptr + offset_bytes,
                single_k_ptr,
                batched_v_ptr + offset_bytes,
                single_v_ptr,
            ));
        }

        // Execute all D2D copies (no borrow conflicts — only raw ptrs)
        for (batched_k_dst, single_k_src, batched_v_dst, single_v_src) in copies {
            self.stream.memcpy_dtod_sync(batched_k_dst, single_k_src, size_bytes)?;
            self.stream.memcpy_dtod_sync(batched_v_dst, single_v_src, size_bytes)?;
        }

        // Update batched KV length for this slot
        if slot_idx < self.batched_kv_lengths.len() {
            self.batched_kv_lengths[slot_idx] = seq_len;
        }

        Ok(())
    }

    /// Clear KV cache for a new generation (reset sequence position to 0)
    pub fn reset_kv_cache_gpu(&mut self) {
        for len in self.kv_cache_lengths.values_mut() {
            *len = 0;
        }
    }

    /// PAR-105: Rollback KV cache to a specific position (for speculative decode)
    ///
    /// This allows undoing speculative tokens without losing the prefill history.
    /// Unlike reset_kv_cache_gpu, this preserves KV values up to `position`.
    pub fn rollback_kv_cache_gpu(&mut self, position: usize) {
        for len in self.kv_cache_lengths.values_mut() {
            if *len > position {
                *len = position;
            }
        }
    }

    /// Debug: Read first N values from KV cache at position 0, layer 0
    pub fn debug_kv_cache_values(
        &self,
        layer_idx: usize,
        is_v: bool,
        n: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let key = if is_v {
            format!("kv_{}_v", layer_idx)
        } else {
            format!("kv_{}_k", layer_idx)
        };
        let buf = self
            .kv_cache_gpu
            .get(&key)
            .ok_or_else(|| GpuError::InvalidParameter(format!("KV cache not found: {}", key)))?;
        let total = buf.len();
        let read_n = n.min(total);
        let mut vals = vec![0.0f32; total];
        buf.copy_to_host(&mut vals)?;
        Ok(vals[..read_n].to_vec())
    }

    /// Debug: Dump KV cache values at a specific position for head 0
    pub fn debug_kv_cache_at_position(
        &self,
        layer_idx: usize,
        position: usize,
        is_v: bool,
        n: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let key = if is_v {
            format!("kv_{}_v", layer_idx)
        } else {
            format!("kv_{}_k", layer_idx)
        };
        let buf = self
            .kv_cache_gpu
            .get(&key)
            .ok_or_else(|| GpuError::InvalidParameter(format!("KV cache not found: {}", key)))?;
        let total = buf.len();
        let mut vals = vec![0.0f32; total];
        buf.copy_to_host(&mut vals)?;
        // KV cache layout: [num_kv_heads, max_len, head_dim]
        // Head 0 starts at offset 0, position p starts at p * head_dim
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let offset = position * head_dim; // head 0
        if offset + n > max_len * head_dim {
            return Ok(vec![]);
        }
        Ok(vals[offset..offset + n.min(head_dim)].to_vec())
    }

    /// PAR-060: Set RoPE theta (rotary position embedding base frequency)
    ///
    /// This must be called after init_kv_cache_gpu with the model's rope_theta value.
    /// Common values: 10000.0 (LLaMA), 1000000.0 (Qwen2, long context models)
    pub fn set_rope_theta(&mut self, theta: f32) {
        self.rope_theta = theta;
    }

    /// CORRECTNESS-011: Set RoPE type (0=NORM adjacent pairs, 2=NEOX split halves)
    ///
    /// Qwen2.5 models use rope_type=2 (NEOX style).
    pub fn set_rope_type(&mut self, rope_type: u32) {
        self.rope_type = rope_type;
    }

    /// PAR-060: Apply RoPE to Q and K vectors (CPU fallback, will be GPU-accelerated later)
    ///
    /// Rotates Q and K by position-dependent angles to inject positional information.
    /// This is called before attention to enable position-aware attention.
    fn apply_rope_to_buffer(&self, buffer: &mut [f32], num_heads: usize, position: usize) {
        let head_dim = self.kv_head_dim;
        let half_dim = head_dim / 2;

        for h in 0..num_heads {
            let head_start = h * head_dim;

            for i in 0..half_dim {
                let freq = 1.0 / self.rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx1 = head_start + i;
                let idx2 = head_start + i + half_dim;

                if idx2 < buffer.len() {
                    let x1 = buffer[idx1];
                    let x2 = buffer[idx2];
                    buffer[idx1] = x1 * cos_val - x2 * sin_val;
                    buffer[idx2] = x1 * sin_val + x2 * cos_val;
                }
            }
        }
    }

    /// Get current KV cache length for a layer
    #[must_use]
    pub fn kv_cache_len(&self, layer_idx: usize) -> usize {
        self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0)
    }

    /// Check if GPU KV cache is initialized (PAR-020)
    #[must_use]
    pub fn has_kv_cache_gpu(&self) -> bool {
        self.kv_cache_max_len > 0
    }

    /// realizr#194: Maximum sequence length the GPU KV cache supports.
    ///
    /// Callers must validate input length against this before forwarding
    /// to prevent overflow and CUDA graph poison.
    #[must_use]
    pub fn max_kv_len(&self) -> usize {
        self.kv_cache_max_len
    }
}
