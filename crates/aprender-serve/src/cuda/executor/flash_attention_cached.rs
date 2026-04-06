impl CudaExecutor {

    /// Append new K/V to GPU cache and run flash attention
    ///
    /// This is the main incremental attention method for autoregressive decoding.
    /// Only the new K/V vectors are transferred to GPU (hidden_dim floats each),
    /// avoiding the O(seq_len × hidden_dim) transfer that was the main bottleneck.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Transformer layer index
    /// * `q` - Query vector for current position [hidden_dim]
    /// * `current_k` - Key vector for current position [hidden_dim]
    /// * `current_v` - Value vector for current position [hidden_dim]
    /// * `output` - Output buffer [hidden_dim]
    ///
    /// # Returns
    ///
    /// New total sequence length after appending
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_cached(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        output: &mut [f32],
    ) -> Result<usize, GpuError> {
        contract_pre_fused_kernel!();
        let num_heads = self.kv_num_heads;
        let head_dim = self.kv_head_dim;
        let hidden_dim = num_heads * head_dim;
        let max_len = self.kv_cache_max_len;

        // Validate dimensions
        if q.len() != hidden_dim || current_k.len() != hidden_dim || current_v.len() != hidden_dim {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-018: dimension mismatch - expected {}, got Q[{}] K[{}] V[{}]",
                hidden_dim,
                q.len(),
                current_k.len(),
                current_v.len()
            )));
        }

        // Get current cache length and check bounds
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        let new_len = cache_len + 1;
        if new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-018: KV cache overflow - max_len={}, trying to add position {}",
                max_len, new_len
            )));
        }

        // Get cache buffer keys
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);

        // Reorganize current_k/v from [hidden_dim] to [num_heads, 1, head_dim]
        // and upload to correct position in GPU cache
        // GPU layout: [num_heads, max_len, head_dim]
        // Position for new data: head * (max_len * head_dim) + cache_len * head_dim
        {
            let k_buf = self.kv_cache_gpu.get_mut(&k_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-018: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;

            // Copy each head's K portion to correct position
            for head in 0..num_heads {
                let src_offset = head * head_dim;
                let dst_offset = head * (max_len * head_dim) + cache_len * head_dim;
                k_buf
                    .copy_from_host_at(&current_k[src_offset..src_offset + head_dim], dst_offset)?;
            }
        }

        {
            let v_buf = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-018: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;

            // Copy each head's V portion to correct position
            for head in 0..num_heads {
                let src_offset = head * head_dim;
                let dst_offset = head * (max_len * head_dim) + cache_len * head_dim;
                v_buf
                    .copy_from_host_at(&current_v[src_offset..src_offset + head_dim], dst_offset)?;
            }
        }

        // Update cache length
        self.kv_cache_lengths.insert(layer_idx, new_len);

        // For GPU-only attention, we need to compact K/V from max_len layout to new_len layout
        // This is necessary because the flash attention kernel expects contiguous seq_len data
        //
        // Current GPU layout: [num_heads, max_len, head_dim] with only new_len positions filled
        // Required layout:    [num_heads, new_len, head_dim] contiguous
        //
        // Options:
        // A) D2D copy to compact buffers (faster than D2H+H2D for long sequences)
        // B) Use padded kernel that handles max_len with actual_len mask (requires kernel change)
        // C) For now: read back and use existing flash_attention_multi_head (baseline)
        //
        // PAR-018 Phase 1: Use compacted read approach for correctness
        // PAR-019 (future): Implement D2D compaction or padded kernel for full GPU residency

        let tensor_size = num_heads * new_len * head_dim;

        // Build Q tensor on CPU: [num_heads, new_len, head_dim]
        // Q is the same for all positions (broadcasting optimization possible in future)
        let mut q_full = vec![0.0f32; tensor_size];
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let gpu_head_offset = head * new_len * head_dim;
            for pos in 0..new_len {
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                q_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&q[head_offset..head_offset + head_dim]);
            }
        }

        // Read compacted K/V from GPU cache
        // Uses new copy_to_host_at for partial reads
        let mut k_data = vec![0.0f32; tensor_size];
        let mut v_data = vec![0.0f32; tensor_size];

        {
            let k_buf = self
                .kv_cache_gpu
                .get(&k_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("KV cache K not found".to_string()))?;
            let v_buf = self
                .kv_cache_gpu
                .get(&v_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("KV cache V not found".to_string()))?;

            for head in 0..num_heads {
                let gpu_head_offset = head * max_len * head_dim;
                let out_head_offset = head * new_len * head_dim;

                // Batch read: read new_len contiguous positions per head
                // This is more efficient than per-position reads
                k_buf.copy_to_host_at(
                    &mut k_data[out_head_offset..out_head_offset + new_len * head_dim],
                    gpu_head_offset,
                )?;
                v_buf.copy_to_host_at(
                    &mut v_data[out_head_offset..out_head_offset + new_len * head_dim],
                    gpu_head_offset,
                )?;
            }
        }

        // Run attention.
        // Flash attention requires seq_len >= head_dim (trueno-gpu AttentionKernel
        // shared memory layout limitation).  Fall back to CPU softmax attention
        // for small seq_len (common during early autoregressive generation).
        let mut output_full = vec![0.0f32; tensor_size];
        if new_len >= head_dim {
            self.flash_attention_multi_head(
                &q_full,
                &k_data,
                &v_data,
                &mut output_full,
                new_len as u32,
                head_dim as u32,
                num_heads as u32,
                true, // causal
            )?;
        } else {
            // CPU fallback: standard scaled dot-product attention per head
            let scale = 1.0 / (head_dim as f32).sqrt();
            for head in 0..num_heads {
                let ho = head * new_len * head_dim;
                for row in 0..new_len {
                    // Compute attention scores: Q[row] · K[col] for col <= row (causal)
                    let mut scores = vec![f32::NEG_INFINITY; new_len];
                    for col in 0..=row {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot +=
                                q_full[ho + row * head_dim + d] * k_data[ho + col * head_dim + d];
                        }
                        scores[col] = dot * scale;
                    }
                    // Softmax
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
                    // Weighted sum of V
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for col in 0..=row {
                            let w = ((scores[col] - max_s).exp()) / exp_sum;
                            acc += w * v_data[ho + col * head_dim + d];
                        }
                        output_full[ho + row * head_dim + d] = acc;
                    }
                }
            }
        }

        // Extract output for last position, reorganize to [hidden_dim]
        let last_pos = new_len - 1;
        for head in 0..num_heads {
            let gpu_offset = head * new_len * head_dim + last_pos * head_dim;
            let out_offset = head * head_dim;
            output[out_offset..out_offset + head_dim]
                .copy_from_slice(&output_full[gpu_offset..gpu_offset + head_dim]);
        }

        Ok(new_len)
    }

    /// PAR-020: True GPU-resident incremental attention for M=1 autoregressive decoding
    ///
    /// Unlike `flash_attention_cached` which does D2H+H2D roundtrips, this method:
    /// 1. Appends new K/V to GPU-resident cache (H2D, small transfer)
    /// 2. Launches IncrementalAttentionKernel directly on GPU buffers
    /// 3. Downloads only the output (D2H, small transfer)
    ///
    /// Target performance: Eliminate ~66 MB/token transfer overhead for TinyLlama
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Transformer layer index
    /// * `q` - Query vector for current position [num_heads, head_dim]
    /// * `current_k` - Key vector for current position [num_heads, head_dim]
    /// * `current_v` - Value vector for current position [num_heads, head_dim]
    /// * `output` - Output buffer [num_heads, head_dim]
    ///
    /// # Returns
    ///
    /// New total sequence length after appending
    #[allow(clippy::too_many_arguments)]
    pub fn incremental_attention_gpu(
        &mut self,
        layer_idx: usize,
        q: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        output: &mut [f32],
    ) -> Result<usize, GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let q_dim = num_heads * head_dim; // Q/output dimension
        let kv_dim = num_kv_heads * head_dim; // K/V dimension (smaller for GQA)
        let max_len = self.kv_cache_max_len;

        // PAR-021 GQA: Q has num_heads dimensions, K/V have num_kv_heads dimensions
        if q.len() != q_dim {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-021: Q dimension mismatch - expected {}, got {}",
                q_dim,
                q.len()
            )));
        }
        if current_k.len() != kv_dim || current_v.len() != kv_dim {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-021: K/V dimension mismatch - expected {}, got K[{}] V[{}]",
                kv_dim,
                current_k.len(),
                current_v.len()
            )));
        }

        // Get current cache length and check bounds
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        let new_len = cache_len + 1;
        if new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-020: KV cache overflow - max_len={}, trying to add position {}",
                max_len, new_len
            )));
        }

        // Get cache buffer keys
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);

        // Append new K/V to GPU cache
        // PAR-021 GQA: Layout is [num_kv_heads, max_len, head_dim]
        {
            let k_buf = self.kv_cache_gpu.get_mut(&k_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-020: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            for kv_head in 0..num_kv_heads {
                let src_offset = kv_head * head_dim;
                let dst_offset = kv_head * (max_len * head_dim) + cache_len * head_dim;
                k_buf
                    .copy_from_host_at(&current_k[src_offset..src_offset + head_dim], dst_offset)?;
            }
        }

        {
            let v_buf = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-020: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            for kv_head in 0..num_kv_heads {
                let src_offset = kv_head * head_dim;
                let dst_offset = kv_head * (max_len * head_dim) + cache_len * head_dim;
                v_buf
                    .copy_from_host_at(&current_v[src_offset..src_offset + head_dim], dst_offset)?;
            }
        }

        // Update cache length
        self.kv_cache_lengths.insert(layer_idx, new_len);

        // Upload Q to GPU (small transfer: num_heads * head_dim floats)
        let mut q_buf = GpuBuffer::<f32>::new(&self.context, q_dim)?;
        q_buf.copy_from_host(q)?;

        // Allocate output buffer (same size as Q)
        let out_buf = GpuBuffer::<f32>::new(&self.context, q_dim)?;

        // Get kernel module (PAR-021: includes n_kv_heads for GQA)
        let kernel_type = KernelType::IncrementalAttention {
            max_seq_len: max_len as u32,
            head_dim: head_dim as u32,
            n_heads: num_heads as u32,
            n_kv_heads: num_kv_heads as u32,
            indirect: false,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let ptx = self.kernels.generate_ptx(&kernel_type);
        let module_key = format!(
            "incremental_attention_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads
        );

        if !self.modules.contains_key(&module_key) {
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(module_key.clone(), module);
        }
        let module = self
            .modules
            .get_mut(&module_key)
            .expect("module just inserted");

        // Get K and V buffer pointers
        let k_buf = self
            .kv_cache_gpu
            .get(&k_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?;
        let v_buf = self
            .kv_cache_gpu
            .get(&v_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?;

        // Launch kernel
        // Grid: (num_heads, 1, 1) - one block per head
        // Block: (32, 1, 1) - one warp per block
        let config = LaunchConfig::grid_2d(num_heads as u32, 1, 32, 1);

        let mut ptr_q = q_buf.as_ptr();
        let mut ptr_k = k_buf.as_ptr();
        let mut ptr_v = v_buf.as_ptr();
        let mut ptr_out = out_buf.as_ptr();
        let mut seq_len_val = new_len as u32;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.compute_stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_q) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_k) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_v) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_len_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Synchronize and download output
        self.compute_stream.synchronize()?;
        out_buf.copy_to_host(output)?;

        Ok(new_len)
    }
}
