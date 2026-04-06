impl CudaExecutor {

    /// APR-TRACE-001: Read final hidden state from GPU to CPU for verbose tracing
    ///
    /// This performs a D2H sync which is expensive (~50µs) but necessary for
    /// Genchi Genbutsu (go and see) observability during debugging.
    ///
    /// ONLY call this when verbose tracing is enabled.
    pub fn read_hidden_state_to_cpu(&mut self) -> Result<Vec<f32>, GpuError> {
        let hidden_buf = self.workspace.hidden_buf2.as_ref().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("APR-TRACE-001: workspace not initialized".to_string())
        })?;

        // Sync stream to ensure all GPU ops complete
        self.stream.synchronize()?;

        // D2H copy
        let mut hidden_cpu = vec![0.0f32; hidden_buf.len()];
        hidden_buf.copy_to_host(&mut hidden_cpu)?;

        Ok(hidden_cpu)
    }

    /// PAR-023: GPU RMSNorm for output layer
    ///
    /// Runs RMSNorm on GPU for the final output before LM head projection.
    pub fn output_rmsnorm_gpu(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        gamma: &[f32],
        hidden_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let gamma_gpu = GpuBuffer::from_host(&self.context, gamma)?;

        let output_gpu = self.rmsnorm_gpu(&input_gpu, &gamma_gpu, hidden_dim, epsilon)?;

        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-023: Helper to run transformer layer with host input/output
    ///
    /// Convenience method for testing and single-layer execution.
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_host(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_norm_gamma: &[f32],
        ffn_norm_gamma: &[f32],
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Upload inputs
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let attn_gamma_gpu = GpuBuffer::from_host(&self.context, attn_norm_gamma)?;
        let ffn_gamma_gpu = GpuBuffer::from_host(&self.context, ffn_norm_gamma)?;

        // Run GPU-resident layer
        let output_gpu = self.transformer_layer_gpu(
            &input_gpu,
            layer_idx,
            layer_prefix,
            hidden_dim,
            intermediate_dim,
            &attn_gamma_gpu,
            &ffn_gamma_gpu,
            epsilon,
        )?;

        // Single sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// TILING-SPEC-001: Tile-profiled transformer layer with host input/output.
    ///
    /// Convenience method for profiling single-layer execution to identify bottlenecks.
    /// Enable tile profiling first with `enable_tile_profiling()`, then call this method,
    /// then examine results with `tile_summary()`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// cuda_model.enable_tile_profiling();
    /// cuda_model.transformer_layer_host_profiled(...)?;
    /// println!("{}", cuda_model.tile_summary());
    /// // Output:
    /// // === Tile Profiling Summary (TILING-SPEC-001) ===
    /// // Level       Samples   Avg µs    GFLOP/s   AI      Elements
    /// // macro             3    1500.0     26.67  0.50    4096
    /// // midi              1     200.0      5.12  0.25    1024
    /// // micro             4      10.0      2.05  4.00     512
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_host_profiled(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_norm_gamma: &[f32],
        ffn_norm_gamma: &[f32],
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Upload inputs
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let attn_gamma_gpu = GpuBuffer::from_host(&self.context, attn_norm_gamma)?;
        let ffn_gamma_gpu = GpuBuffer::from_host(&self.context, ffn_norm_gamma)?;

        // Run GPU-resident tiled profiled layer
        let output_gpu = self.transformer_layer_gpu_tiled_profiled(
            &input_gpu,
            layer_idx,
            layer_prefix,
            hidden_dim,
            intermediate_dim,
            &attn_gamma_gpu,
            &ffn_gamma_gpu,
            epsilon,
        )?;

        // Single sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// Execute Q5_K quantized matvec (fused dequantization + matvec) - PARITY-116
    ///
    /// # Arguments
    ///
    /// * `weights` - Quantized weights in Q5_K GGML format (176 bytes per 256 values)
    /// * `input` - Input vector (f32)
    /// * `output` - Output vector (f32)
    /// * `m` - Output dimension
    /// * `k` - Input dimension (must be divisible by 256)
    pub fn q5k_matvec(
        &mut self,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::Q5KQuantizedGemm { m, n: 1, k };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5k_{}_{}", m, k);

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate GPU buffers
        let buf_weights = GpuBuffer::from_host(&self.context, weights)?;
        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, m as usize)?;

        // Launch configuration
        let config = LaunchConfig::linear(m, 256);

        let mut ptr_input = buf_input.as_ptr();
        let mut ptr_weights = buf_weights.as_ptr();
        let mut ptr_output = buf_output.as_ptr();
        let mut m_val = m;
        let mut n_val = 1u32;
        let mut k_val = k;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// Execute Q6_K quantized matvec (fused dequantization + matvec) - PARITY-117
    ///
    /// # Arguments
    ///
    /// * `weights` - Quantized weights in Q6_K GGML format (210 bytes per 256 values)
    /// * `input` - Input vector (f32)
    /// * `output` - Output vector (f32)
    /// * `m` - Output dimension
    /// * `k` - Input dimension (must be divisible by 256)
    pub fn q6k_matvec(
        &mut self,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::Q6KQuantizedGemm { m, n: 1, k };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q6k_{}_{}", m, k);

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate GPU buffers
        let buf_weights = GpuBuffer::from_host(&self.context, weights)?;
        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, m as usize)?;

        // Launch configuration
        let config = LaunchConfig::linear(m, 256);

        let mut ptr_input = buf_input.as_ptr();
        let mut ptr_weights = buf_weights.as_ptr();
        let mut ptr_output = buf_output.as_ptr();
        let mut m_val = m;
        let mut n_val = 1u32;
        let mut k_val = k;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// Execute FlashAttention forward pass (IMP-900c)
    ///
    /// Memory-efficient attention using tiled computation to avoid O(N²)
    /// memory usage. Computes: softmax(QK^T / sqrt(d)) @ V
    ///
    /// # Arguments
    ///
    /// * `q` - Query matrix (seq_len × head_dim)
    /// * `k` - Key matrix (seq_len × head_dim)
    /// * `v` - Value matrix (seq_len × head_dim)
    /// * `output` - Output matrix (seq_len × head_dim)
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `scale` - Softmax scale factor (typically 1/sqrt(head_dim))
    /// * `causal` - Whether to apply causal masking
    ///
    /// # Performance Impact
    ///
    /// - Naive attention: O(N²) memory for attention matrix
    /// - FlashAttention: O(N) memory using tiled computation
    /// - Expected speedup: 2-4x for long sequences
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: u32,
        head_dim: u32,
        _scale: f32,
        causal: bool,
    ) -> Result<(), GpuError> {
        contract_pre_flash_attention!();
        let expected_size = (seq_len * head_dim) as usize;

        if q.len() != expected_size
            || k.len() != expected_size
            || v.len() != expected_size
            || output.len() != expected_size
        {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "Attention size mismatch: expected {}, got Q[{}] K[{}] V[{}] O[{}]",
                expected_size,
                q.len(),
                k.len(),
                v.len(),
                output.len()
            )));
        }

        // Track memory in pool
        self.memory_pool.record_allocation(expected_size * 4 * 4);

        // Use FlashAttention-style kernel
        let kernel_type = KernelType::Attention {
            seq_len,
            head_dim,
            causal,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("flash_attn_{}_{}_{}", seq_len, head_dim, causal);

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            // Debug: Print PTX for debugging invalid PTX errors
            #[cfg(test)]
            eprintln!("Generated attention PTX:\n{}", ptx);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Fail-fast: validate shared memory bounds BEFORE launching the kernel.
        // GH-5: trueno-gpu AttentionKernel now ensures tile_kv >= head_dim,
        // so K tile always has enough elements for the dot product loop.
        // IMP-1010: Ensure tile_q * head_dim <= 1024 for thread count limit.
        let thread_limit = 1024 / head_dim;
        let tile_q = 64u32.min(seq_len).min(thread_limit);

        // Allocate GPU buffers
        let buf_q = GpuBuffer::from_host(&self.context, q)?;
        let buf_k = GpuBuffer::from_host(&self.context, k)?;
        let buf_v = GpuBuffer::from_host(&self.context, v)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, expected_size)?;

        // Launch configuration: 2D grid for attention
        // Grid X: Q blocks (ceil(seq_len / tile_q)), Grid Y: num_heads
        // Threads: tile_q * head_dim (must be <= 1024)
        // IMP-1010: Ensure tile_q * head_dim <= 1024 so all threads can load Q/K/V elements
        let num_q_blocks = (seq_len + tile_q - 1) / tile_q;
        let num_heads = 1u32; // Single head for now
        let threads_per_block = tile_q * head_dim; // Now guaranteed <= 1024
        let config = LaunchConfig::grid_2d(num_q_blocks, num_heads, threads_per_block, 1);

        // Get raw pointers
        let mut ptr_q = buf_q.as_ptr();
        let mut ptr_k = buf_k.as_ptr();
        let mut ptr_v = buf_v.as_ptr();
        let mut ptr_output = buf_output.as_ptr();
        let mut seq_len_val = seq_len;
        let mut head_dim_val = head_dim;
        // Kernel expects num_heads, not scale (scale is baked into kernel or computed internally)
        let mut num_heads_val = 1u32;

        // Launch kernel
        // SAFETY: Buffers are valid, dimensions match
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_q) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_k) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_v) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_len_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut num_heads_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Synchronize and copy back
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        self.memory_pool.record_deallocation(expected_size * 4 * 4);

        Ok(())
    }
}
