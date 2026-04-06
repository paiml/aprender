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
    /// PMAT-399: Compute maximum batch size that fits in available GPU memory.
    /// GH-178: Compute max batch size that fits in available VRAM.
    ///
    /// Reserves space for:
    /// - FP8/FP16 prefill weight cache (~1.5-3GB for 1.5B model)
    /// - cuBLAS workspace (32MB)
    /// - CUDA runtime overhead (~500MB)
    ///
    /// Without this, the server starts but OOMs on first request
    /// when cohabiting GPU with other processes (e.g. training).
    pub fn compute_max_batch_for_memory(
        &self,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_len: usize,
    ) -> usize {
        let (free, _total) = self.context.memory_info().unwrap_or((8 * 1024 * 1024 * 1024, 0));
        // KV cache per slot: 2 (K+V) × num_kv_heads × max_len × head_dim × 4 bytes × num_layers
        let kv_per_slot = 2 * num_kv_heads * max_len * head_dim * 4 * num_layers;

        // GH-178: Reserve VRAM for prefill cache + workspace.
        // FP8 cache: ~N_params × 1 byte (1.5GB for 1.5B model)
        // FP16 cache: ~N_params × 2 bytes (3GB for 1.5B model)
        // cuBLAS workspace: 32MB
        // CUDA runtime overhead: ~500MB
        //
        // Conservative: assume FP16 (larger) + 1GB headroom.
        // This prevents silent OOM when GPU is shared.
        let prefill_cache_estimate = if self.gpu_profile.fp8_prefill {
            // FP8: ~1 byte per weight element
            num_layers * num_kv_heads * head_dim * 16 * 1 // rough estimate
        } else {
            0
        };
        // Use 3.5GB as conservative reserve (was 2GB)
        // This covers FP8(1.5GB) + workspace(32MB) + runtime(500MB) + headroom
        let reserve = (3_500_000_000_usize).max(prefill_cache_estimate + 512 * 1024 * 1024);
        let available = free.saturating_sub(reserve);

        let max_batch = if kv_per_slot > 0 { available / kv_per_slot } else { 32 };
        let clamped = max_batch.clamp(1, 32);
        // GH-611: Suppressed — was noisy in non-verbose mode
        if clamped <= 1 && free < reserve {
            eprintln!(
                "[GH-178] WARNING: Only {:.1} GB free VRAM (need {:.1} GB \
                 for prefill cache + workspace). Inference may OOM. \
                 Free GPU memory or use --device cpu.",
                free as f64 / 1e9, reserve as f64 / 1e9,
            );
        }
        clamped
    }

    /// Initialize per-layer KV cache on GPU for single-sequence inference.
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

        // PMAT-075: Skip auxiliary buffer reallocation when KV caches are preserved.
        // The captured batched decode graph holds pointers to batched_k_ptrs,
        // batched_v_ptrs, batched_seq_lens_gpu, and per-layer pointer buffers.
        // Reallocating these gives new addresses → stale graph → ILLEGAL_ADDRESS.
        // When !need_realloc, KV cache buffers haven't changed, so per-layer
        // pointer buffers still hold correct addresses.
        if !need_realloc
            && self.batched_k_ptrs.is_some()
            && self.batched_v_ptrs.is_some()
            && self.batched_seq_lens_gpu.is_some()
            && self.batched_k_ptrs_per_layer.len() == num_layers
        {
            eprintln!(
                "[PMAT-075] Reusing batched KV cache: {} layers × {} sequences (addresses stable)",
                num_layers, batch_size
            );
            return Ok(());
        }

        // Allocate GPU pointer arrays for batched attention
        self.batched_k_ptrs = Some(GpuBuffer::new(&self.context, batch_size)?);
        self.batched_v_ptrs = Some(GpuBuffer::new(&self.context, batch_size)?);
        self.batched_seq_lens_gpu = Some(GpuBuffer::new(&self.context, batch_size)?);

        // GH-141: Pre-populate per-layer pointer buffers for CUDA graph capture.
        // During graph capture, H2D copies are not capturable, so we can't update
        // the shared batched_k_ptrs per layer. These per-layer buffers contain
        // static KV cache base addresses that the graph records directly.
        self.batched_k_ptrs_per_layer.clear();
        self.batched_v_ptrs_per_layer.clear();
        let stride_bytes = (stride * std::mem::size_of::<f32>()) as u64;
        for layer_idx in 0..num_layers {
            if let (Some(k_cache), Some(v_cache)) = (
                self.batched_kv_k_caches.get(&layer_idx),
                self.batched_kv_v_caches.get(&layer_idx),
            ) {
                let k_ptrs: Vec<u64> = (0..batch_size)
                    .map(|i| k_cache.as_ptr() + i as u64 * stride_bytes)
                    .collect();
                let v_ptrs: Vec<u64> = (0..batch_size)
                    .map(|i| v_cache.as_ptr() + i as u64 * stride_bytes)
                    .collect();
                self.batched_k_ptrs_per_layer
                    .insert(layer_idx, GpuBuffer::from_host(&self.context, &k_ptrs)?);
                self.batched_v_ptrs_per_layer
                    .insert(layer_idx, GpuBuffer::from_host(&self.context, &v_ptrs)?);
            }
        }

        // PMAT-075: Auxiliary buffer reallocation invalidates captured batched graphs.
        self.batched_decode_graphs.clear();
        self.batched_graph_batch_size = 0;

        let total_bytes = num_layers * 2 * buffer_size * 4 + batch_size * 24
            + num_layers * 2 * batch_size * 8; // caches + ptr arrays + per-layer ptrs
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

    /// PMAT-051: Copy single KV cache to batched KV cache for ONE layer.
    ///
    /// Used during multi-prompt batched prefill: within the layer loop,
    /// each prompt's KV is prefilled into the single cache, then scattered
    /// to the batched slot for that layer only (not all layers).
    pub fn scatter_single_kv_to_batched_layer(
        &mut self,
        slot_idx: usize,
        seq_len: usize,
        layer_idx: usize,
    ) -> Result<(), GpuError> {
        if seq_len == 0 {
            return Ok(());
        }

        let stride = self.batched_kv_stride;
        if stride == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "PMAT-051: batched KV cache not initialized (stride=0)".to_string(),
            ));
        }

        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let per_head_copy_bytes = (seq_len * head_dim * std::mem::size_of::<f32>()) as u64;
        let head_stride_bytes = (max_len * head_dim * std::mem::size_of::<f32>()) as u64;
        let slot_offset_bytes = (slot_idx * stride * std::mem::size_of::<f32>()) as u64;

        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);

        let single_k_ptr = self
            .kv_cache_gpu
            .get(&k_key)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PMAT-051: single KV cache '{}' not found",
                    k_key
                ))
            })?
            .as_ptr();
        let batched_k_ptr = self
            .batched_kv_k_caches
            .get(&layer_idx)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PMAT-051: batched K cache layer {} not found",
                    layer_idx
                ))
            })?
            .as_ptr();

        let single_v_ptr = self
            .kv_cache_gpu
            .get(&v_key)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PMAT-051: single KV cache '{}' not found",
                    v_key
                ))
            })?
            .as_ptr();
        let batched_v_ptr = self
            .batched_kv_v_caches
            .get(&layer_idx)
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PMAT-051: batched V cache layer {} not found",
                    layer_idx
                ))
            })?
            .as_ptr();

        for head in 0..num_kv_heads as u64 {
            let head_off = head * head_stride_bytes;
            self.stream.memcpy_dtod_sync(
                batched_k_ptr + slot_offset_bytes + head_off,
                single_k_ptr + head_off,
                per_head_copy_bytes as usize,
            )?;
            self.stream.memcpy_dtod_sync(
                batched_v_ptr + slot_offset_bytes + head_off,
                single_v_ptr + head_off,
                per_head_copy_bytes as usize,
            )?;
        }

        // Update batched KV length for this slot
        if slot_idx < self.batched_kv_lengths.len() {
            self.batched_kv_lengths[slot_idx] = seq_len;
        }

        Ok(())
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
        let max_len = self.kv_cache_max_len;
        // Per-head copy size (only the filled positions, not full max_len)
        let per_head_copy_bytes = (seq_len * head_dim * std::mem::size_of::<f32>()) as u64;
        // Per-head stride in bytes (full max_len allocation per head)
        let head_stride_bytes = (max_len * head_dim * std::mem::size_of::<f32>()) as u64;
        let slot_offset_bytes = (slot_idx * stride * std::mem::size_of::<f32>()) as u64;

        // Collect pointer pairs to avoid borrow conflicts between HashMap fields
        let layer_indices: Vec<usize> = self.batched_kv_k_caches.keys().copied().collect();
        let mut copies: Vec<(u64, u64, u64, u64)> = Vec::new();

        for &layer_idx in &layer_indices {
            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);

            let single_k_ptr = self.kv_cache_gpu.get(&k_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig(
                    format!("PMAT-044: single KV cache '{}' not found", k_key)
                ))?.as_ptr();
            let batched_k_ptr = self.batched_kv_k_caches.get(&layer_idx)
                .ok_or_else(|| GpuError::InvalidLaunchConfig(
                    format!("PMAT-044: batched K cache layer {} not found", layer_idx)
                ))?.as_ptr();

            let single_v_ptr = self.kv_cache_gpu.get(&v_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig(
                    format!("PMAT-044: single KV cache '{}' not found", v_key)
                ))?.as_ptr();
            let batched_v_ptr = self.batched_kv_v_caches.get(&layer_idx)
                .ok_or_else(|| GpuError::InvalidLaunchConfig(
                    format!("PMAT-044: batched V cache layer {} not found", layer_idx)
                ))?.as_ptr();

            copies.push((
                batched_k_ptr + slot_offset_bytes,
                single_k_ptr,
                batched_v_ptr + slot_offset_bytes,
                single_v_ptr,
            ));
        }

        // Copy per-head: layout is [num_kv_heads, max_len, head_dim]
        // Each head's data is head_stride_bytes apart, copy only seq_len positions
        for (k_dst, k_src, v_dst, v_src) in copies {
            for head in 0..num_kv_heads as u64 {
                let head_off = head * head_stride_bytes;
                self.stream.memcpy_dtod_sync(
                    k_dst + head_off, k_src + head_off, per_head_copy_bytes as usize,
                )?;
                self.stream.memcpy_dtod_sync(
                    v_dst + head_off, v_src + head_off, per_head_copy_bytes as usize,
                )?;
            }
        }

        // Update batched KV length for this slot
        if slot_idx < self.batched_kv_lengths.len() {
            self.batched_kv_lengths[slot_idx] = seq_len;
        }

        Ok(())
    }

    /// PMAT-058: Free batched KV caches to reclaim VRAM after batch decode.
    ///
    /// Five-Whys: c=1 decode regresses 140→124 tok/s after c=4 batch.
    /// Why? SGEMM prefill (no FP16 cache) instead of HGEMM.
    /// Why? FP16 weight cache was cleared before batch decode (GH-141).
    /// Why? Not rebuilt because batched KV caches (~460MB) still occupy VRAM.
    /// Why? generate_batched_streaming didn't free them after decode.
    /// Fix: Free all batched KV state so FP16 cache can be rebuilt on next c=1.
    pub fn free_batched_kv_caches(&mut self) {
        let had_caches = !self.batched_kv_k_caches.is_empty();
        self.batched_kv_k_caches.clear();
        self.batched_kv_v_caches.clear();
        self.batched_k_ptrs_per_layer.clear();
        self.batched_v_ptrs_per_layer.clear();
        self.batched_k_ptrs = None;
        self.batched_v_ptrs = None;
        self.batched_seq_lens_gpu = None;
        self.batched_kv_lengths.clear();
        self.batched_kv_allocated_batch = 0;
        if had_caches {
            eprintln!("[PMAT-058] Freed batched KV caches to reclaim VRAM for FP16 rebuild");
        }
    }

    /// Clear KV cache for a new generation (reset sequence position to 0)
    pub fn reset_kv_cache_gpu(&mut self) {
        for len in self.kv_cache_lengths.values_mut() {
            *len = 0;
        }
    }

    /// CORRECTNESS-016: Zero-fill all KV cache buffers (diagnostic).
    /// Used to distinguish "scatter didn't write" (zeros) from "scatter wrote wrong values".
    pub fn zero_kv_cache_gpu(&mut self) -> Result<(), GpuError> {
        for buf in self.kv_cache_gpu.values_mut() {
            let zeros = vec![0.0f32; buf.len()];
            buf.copy_from_host(&zeros)?;
        }
        for len in self.kv_cache_lengths.values_mut() {
            *len = 0;
        }
        Ok(())
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

    /// CORRECTNESS-016: Per-position sum fingerprint of L0 K cache head 0.
    /// Returns one f32 per position (sum of head_dim elements).
    pub fn kv_cache_l0_k_fingerprint(&self, num_positions: usize) -> Result<Vec<f32>, GpuError> {
        let key = "kv_0_k".to_string();
        let buf = self
            .kv_cache_gpu
            .get(&key)
            .ok_or_else(|| GpuError::InvalidParameter("kv_0_k not found".to_string()))?;
        let mut vals = vec![0.0f32; buf.len()];
        buf.copy_to_host(&mut vals)?;
        let head_dim = self.kv_head_dim;
        Ok((0..num_positions)
            .map(|p| {
                let start = p * head_dim;
                let end = (start + head_dim).min(vals.len());
                if start < vals.len() {
                    vals[start..end].iter().sum::<f32>()
                } else {
                    0.0
                }
            })
            .collect())
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

    /// realizr#199 (PMAT-450): Set KV cache length for a specific layer.
    /// Used to temporarily truncate cache for prompt-only snapshot.
    pub fn set_kv_cache_len(&mut self, layer_idx: usize, len: usize) {
        self.kv_cache_lengths.insert(layer_idx, len);
    }

    /// Check if GPU KV cache is initialized (PAR-020)
    #[must_use]
    pub fn has_kv_cache_gpu(&self) -> bool {
        self.kv_cache_max_len > 0
    }

    /// realizr#194: Maximum sequence length the GPU KV cache supports.
    ///
    /// Callers must validate input length against this before forwarding
    /// to prevent KV overflow and CUDA state poisoning.
    #[must_use]
    pub fn max_kv_len(&self) -> usize {
        self.kv_cache_max_len
    }

    /// realizr#199 (PMAT-450): Copy GPU KV cache to host for prefix caching.
    ///
    /// Returns per-layer (K, V) vectors covering positions 0..seq_len.
    /// Layout per layer: flattened [num_kv_heads × seq_len × head_dim].
    pub fn snapshot_kv_cache_to_host(
        &mut self,
        num_layers: usize,
    ) -> Result<Vec<(Vec<f32>, Vec<f32>)>, GpuError> {
        self.stream.synchronize()?;
        let mut result = Vec::with_capacity(num_layers);
        let kv_dim = self.kv_num_kv_heads * self.kv_head_dim;

        for layer_idx in 0..num_layers {
            let seq_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
            let copy_elements = kv_dim * seq_len;

            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);

            let mut k_host = vec![0.0f32; copy_elements];
            let mut v_host = vec![0.0f32; copy_elements];

            if copy_elements > 0 {
                // KV layout is [num_kv_heads, max_len, head_dim].
                // For prefix cache we need contiguous [num_kv_heads, seq_len, head_dim].
                // Since max_len may differ from seq_len, copy per-head slices.
                let k_buf = self.kv_cache_gpu.get(&k_key).ok_or_else(|| {
                    GpuError::InvalidParameter(format!("KV cache not found: {}", k_key))
                })?;
                let v_buf = self.kv_cache_gpu.get(&v_key).ok_or_else(|| {
                    GpuError::InvalidParameter(format!("KV cache not found: {}", v_key))
                })?;

                // Full D2H then extract (simpler than per-head strided copy)
                let total = k_buf.len();
                let mut k_full = vec![0.0f32; total];
                let mut v_full = vec![0.0f32; total];
                k_buf.copy_to_host(&mut k_full)?;
                v_buf.copy_to_host(&mut v_full)?;

                // Extract [num_kv_heads, seq_len, head_dim] from [num_kv_heads, max_len, head_dim]
                let max_len = self.kv_cache_max_len;
                let head_dim = self.kv_head_dim;
                for head in 0..self.kv_num_kv_heads {
                    for pos in 0..seq_len {
                        let src_offset = head * max_len * head_dim + pos * head_dim;
                        let dst_offset = head * seq_len * head_dim + pos * head_dim;
                        k_host[dst_offset..dst_offset + head_dim]
                            .copy_from_slice(&k_full[src_offset..src_offset + head_dim]);
                        v_host[dst_offset..dst_offset + head_dim]
                            .copy_from_slice(&v_full[src_offset..src_offset + head_dim]);
                    }
                }
            }

            result.push((k_host, v_host));
        }

        Ok(result)
    }

    /// realizr#199 (PMAT-450): Restore GPU KV cache from host snapshot.
    ///
    /// Copies per-layer (K, V) vectors into GPU buffers and sets cache lengths.
    /// Used to skip prefill when a prompt prefix cache hits.
    pub fn restore_kv_cache_from_host(
        &mut self,
        kv_data: &[(Vec<f32>, Vec<f32>)],
        seq_len: usize,
    ) -> Result<(), GpuError> {
        let kv_dim = self.kv_num_kv_heads * self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let head_dim = self.kv_head_dim;

        if seq_len > max_len {
            return Err(GpuError::InvalidParameter(format!(
                "PMAT-450: seq_len {} > max_len {}", seq_len, max_len
            )));
        }

        for (layer_idx, (k_host, v_host)) in kv_data.iter().enumerate() {
            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);

            // Expand [num_kv_heads, seq_len, head_dim] → [num_kv_heads, max_len, head_dim]
            let buf_len = self.kv_cache_gpu.get(&k_key)
                .ok_or_else(|| GpuError::InvalidParameter(format!("KV cache not found: {}", k_key)))?
                .len();
            let mut k_full = vec![0.0f32; buf_len];
            let mut v_full = vec![0.0f32; buf_len];

            for head in 0..self.kv_num_kv_heads {
                for pos in 0..seq_len {
                    let src_offset = head * seq_len * head_dim + pos * head_dim;
                    let dst_offset = head * max_len * head_dim + pos * head_dim;
                    if src_offset + head_dim <= k_host.len() {
                        k_full[dst_offset..dst_offset + head_dim]
                            .copy_from_slice(&k_host[src_offset..src_offset + head_dim]);
                        v_full[dst_offset..dst_offset + head_dim]
                            .copy_from_slice(&v_host[src_offset..src_offset + head_dim]);
                    }
                }
            }

            let k_buf = self.kv_cache_gpu.get_mut(&k_key).expect("just checked");
            k_buf.copy_from_host(&k_full)?;
            let v_buf = self.kv_cache_gpu.get_mut(&v_key).expect("just checked");
            v_buf.copy_from_host(&v_full)?;
            self.kv_cache_lengths.insert(layer_idx, seq_len);
        }

        Ok(())
    }
}
