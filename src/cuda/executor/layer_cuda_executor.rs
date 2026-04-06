
impl CudaExecutor {

    /// Execute multi-head FlashAttention forward pass (PARITY-043)
    ///
    /// Processes all attention heads in parallel for maximum GPU occupancy.
    /// Each CUDA block handles one attention head independently.
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor [n_heads, seq_len, head_dim]
    /// * `k` - Key tensor [n_heads, seq_len, head_dim]
    /// * `v` - Value tensor [n_heads, seq_len, head_dim]
    /// * `output` - Output tensor [n_heads, seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Dimension per head (typically 64 or 128)
    /// * `n_heads` - Number of attention heads to process in parallel
    /// * `causal` - Whether to apply causal masking (autoregressive)
    ///
    /// # Performance
    ///
    /// - Parallelization: n_heads blocks × seq_len threads
    /// - Memory: O(n_heads × seq_len × head_dim) for K/V shared memory
    /// - Expected speedup: ~n_heads× over sequential single-head attention
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_multi_head(
        &mut self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: u32,
        head_dim: u32,
        n_heads: u32,
        causal: bool,
    ) -> Result<(), GpuError> {
        let head_size = (seq_len * head_dim) as usize;
        let total_size = head_size * n_heads as usize;

        // Validate input sizes
        if q.len() != total_size
            || k.len() != total_size
            || v.len() != total_size
            || output.len() != total_size
        {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "Multi-head attention size mismatch: expected {} ({}×{}×{}), got Q[{}] K[{}] V[{}] O[{}]",
                total_size, n_heads, seq_len, head_dim,
                q.len(), k.len(), v.len(), output.len()
            )));
        }

        // Track memory allocation
        self.memory_pool.record_allocation(total_size * 4 * 4);

        // Generate/cache multi-head attention kernel
        let kernel_type = KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!(
            "multi_head_attn_{}_{}_{}_{}",
            seq_len, head_dim, n_heads, causal
        );

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            #[cfg(test)]
            eprintln!("Generated multi-head attention PTX:\n{}", ptx);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate GPU buffers
        let buf_q = GpuBuffer::from_host(&self.context, q)?;
        let buf_k = GpuBuffer::from_host(&self.context, k)?;
        let buf_v = GpuBuffer::from_host(&self.context, v)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, total_size)?;

        // Launch configuration for trueno's FlashAttention kernel:
        // - Grid.x = number of Q tile blocks (ceil(seq_len / tile_q))
        // - Grid.y = number of heads
        // - Threads = tile_q * head_dim (each thread handles one element)
        // Calculate tile size to fit in 48KB shared memory (same as generate_ptx)
        let max_tile = (48 * 1024) / (head_dim * 12);
        // IMP-1010: Ensure tile_q * head_dim <= 1024 so all threads can load Q/K/V elements
        // Without this constraint, we launch 1024 threads but need tile_q * head_dim > 1024 loads
        let thread_limit = 1024 / head_dim;
        let tile_q = max_tile.min(64).min(seq_len).min(thread_limit);
        // GH-5 FIX: Ensure tile_kv >= head_dim to match trueno-gpu AttentionKernel fix
        // The K dot product loop accesses K[local_col * head_dim + d_idx], requiring
        // head_dim rows in K tile. This is now fixed in trueno-gpu.
        let _tile_kv = seq_len.min(64).max(head_dim);

        let num_q_blocks = (seq_len + tile_q - 1) / tile_q;
        let threads_per_block = tile_q * head_dim; // Now guaranteed <= 1024
        let config = LaunchConfig::grid_2d(num_q_blocks, n_heads, threads_per_block, 1);

        // Get raw pointers for kernel args
        let mut ptr_q = buf_q.as_ptr();
        let mut ptr_k = buf_k.as_ptr();
        let mut ptr_v = buf_v.as_ptr();
        let mut ptr_output = buf_output.as_ptr();
        let mut seq_len_val = seq_len;
        let mut head_dim_val = head_dim;
        let mut n_heads_val = n_heads;

        // Launch kernel
        // SAFETY: Buffers are valid, dimensions match, pointers are aligned
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
                    std::ptr::from_mut(&mut n_heads_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Synchronize and copy back
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        self.memory_pool.record_deallocation(total_size * 4 * 4);

        Ok(())
    }
}

include!("swiglu.rs");
include!("cached.rs");
include!("layer_trace_and_output_norm.rs");
