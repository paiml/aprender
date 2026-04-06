impl CudaExecutor {
    // =========================================================================
    // PAR-023: GPU-Resident Incremental Attention (No Sync)
    // Reduces sync per attention call by keeping Q/K/V on GPU
    // =========================================================================

    /// PAR-023: GPU-resident incremental attention operating on GPU buffers
    ///
    /// Same as `incremental_attention_gpu` but takes GPU buffers instead of
    /// host slices, allowing full GPU pipeline without intermediate syncs.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index for KV cache lookup
    /// * `q_gpu` - Query GPU buffer [num_heads * head_dim]
    /// * `k_gpu` - Current key GPU buffer [num_kv_heads * head_dim]
    /// * `v_gpu` - Current value GPU buffer [num_kv_heads * head_dim]
    ///
    /// # Returns
    /// (output_gpu, new_seq_len) - Attention output buffer and updated sequence length
    #[allow(clippy::too_many_arguments)]
    pub fn incremental_attention_async(
        &mut self,
        layer_idx: usize,
        q_gpu: &GpuBuffer<f32>,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
    ) -> Result<(GpuBuffer<f32>, usize), GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let q_dim = num_heads * head_dim;
        let max_len = self.kv_cache_max_len;

        // Get current cache length and check bounds
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        let new_len = cache_len + 1;
        if new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-023: KV cache overflow - max_len={}, trying to add position {}",
                max_len, new_len
            )));
        }

        // Get cache buffer keys
        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);

        // PAR-023: Copy K/V from GPU buffers to cache positions (D2D transfer)
        // Layout is [num_kv_heads, max_len, head_dim]
        // We need to copy each head's current K/V to the correct position
        //
        // Using D2D copy to avoid host round-trip (zero-sync attention)
        {
            let k_buf = self.kv_cache_gpu.get_mut(&k_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            for kv_head in 0..num_kv_heads {
                let src_offset = kv_head * head_dim;
                let dst_offset = kv_head * (max_len * head_dim) + cache_len * head_dim;
                k_buf.copy_from_buffer_at(k_gpu, dst_offset, src_offset, head_dim)?;
            }

            let v_buf = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            for kv_head in 0..num_kv_heads {
                let src_offset = kv_head * head_dim;
                let dst_offset = kv_head * (max_len * head_dim) + cache_len * head_dim;
                v_buf.copy_from_buffer_at(v_gpu, dst_offset, src_offset, head_dim)?;
            }
        }

        // Update cache length
        self.kv_cache_lengths.insert(layer_idx, new_len);

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

        // Get K and V buffer pointers from cache
        let k_buf = self
            .kv_cache_gpu
            .get(&k_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?;
        let v_buf = self
            .kv_cache_gpu
            .get(&v_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?;

        // Launch kernel
        let config = LaunchConfig::grid_2d(num_heads as u32, 1, 32, 1);

        let mut ptr_q = q_gpu.as_ptr();
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

        // PAR-023: NO sync here - caller continues pipeline
        Ok((out_buf, new_len))
    }

    /// PAR-051: Incremental attention writing into pre-allocated output buffer
    ///
    /// Like `incremental_attention_async` but eliminates GPU allocation by
    /// writing directly into the provided output buffer.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Transformer layer index (for KV cache lookup)
    /// * `q_gpu` - Query tensor on GPU [q_dim]
    /// * `k_gpu` - Key tensor on GPU [kv_dim] (will be appended to cache)
    /// * `v_gpu` - Value tensor on GPU [kv_dim] (will be appended to cache)
    /// * `out_gpu` - Pre-allocated output buffer [q_dim]
    ///
    /// # Returns
    ///
    /// New sequence length after appending K/V to cache
    pub fn incremental_attention_into(
        &mut self,
        layer_idx: usize,
        q_gpu: &GpuBuffer<f32>,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
        out_gpu: &GpuBuffer<f32>,
    ) -> Result<usize, GpuError> {
        self.incremental_attention_into_inner(layer_idx, q_gpu, k_gpu, v_gpu, out_gpu, false)
    }

    /// PAR-054-FIX: Version for graph capture that skips debug sync/copy
    pub(crate) fn incremental_attention_into_for_capture(
        &mut self,
        layer_idx: usize,
        q_gpu: &GpuBuffer<f32>,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
        out_gpu: &GpuBuffer<f32>,
    ) -> Result<usize, GpuError> {
        self.incremental_attention_into_inner(layer_idx, q_gpu, k_gpu, v_gpu, out_gpu, true)
    }
}
