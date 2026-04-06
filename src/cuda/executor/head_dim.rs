impl CudaExecutor {

    /// PAR-118: Flash Decoding attention using split-K parallelism.
    ///
    /// Splits the KV cache into chunks and processes them in parallel,
    /// then reduces partial results with proper softmax rescaling.
    ///
    /// # Arguments
    /// * `layer_idx` - Transformer layer index
    /// * `q_batched` - Q projections [M, num_heads, head_dim]
    /// * `k_batched` - K projections [M, num_kv_heads, head_dim]
    /// * `v_batched` - V projections [M, num_kv_heads, head_dim]
    /// * `out_batched` - Output buffer [M, num_heads, head_dim]
    /// * `m` - Batch size (number of sequences)
    /// * `positions` - Position for each sequence [M]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_decoding_attention_into(
        &mut self,
        layer_idx: usize,
        q_batched: &GpuBuffer<f32>,
        k_batched: &GpuBuffer<f32>,
        v_batched: &GpuBuffer<f32>,
        out_batched: &GpuBuffer<f32>,
        m: usize,
        positions: &[u32],
    ) -> Result<(), GpuError> {
        use trueno_gpu::kernels::{
            FlashDecodingChunkKernel, FlashDecodingReduceKernel, Kernel, FLASH_DECODE_CHUNK_SIZE,
        };

        if !self.flash_decode_enabled {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-118: Flash Decoding not initialized (call init_flash_decoding first)"
                    .to_string(),
            ));
        }

        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let stride = self.batched_kv_stride;

        // Step 1: Scatter K/V to caches (same as batched_incremental_attention_into)
        let kv_dim = num_kv_heads * head_dim;
        let scatter_config = LaunchConfig {
            grid: (num_kv_heads as u32, 1, 1),
            block: (head_dim as u32, 1, 1),
            shared_mem: 0,
        };

        let scatter_type = KernelType::KvCacheScatter {
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            max_len: max_len as u32,
        };
        let scatter_name = self.kernels.kernel_name(&scatter_type);
        let scatter_key = format!("kv_scatter_{}_{}", num_kv_heads, head_dim);

        if !self.modules.contains_key(&scatter_key) {
            let scatter_ptx = self.kernels.generate_ptx(&scatter_type);
            let module = self.compile_ptx(&scatter_ptx)?;
            self.modules.insert(scatter_key.clone(), module);
        }

        let k_cache = self.batched_kv_k_caches.get(&layer_idx).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "PAR-118: Batched K cache not found for layer {}",
                layer_idx
            ))
        })?;
        let v_cache = self.batched_kv_v_caches.get(&layer_idx).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "PAR-118: Batched V cache not found for layer {}",
                layer_idx
            ))
        })?;

        // Scatter K and V for each sequence
        for seq_idx in 0..m {
            let pos = positions[seq_idx] as usize;

            let k_src_offset = seq_idx * kv_dim;
            let k_dst_offset = seq_idx * stride;
            let k_src_ptr = k_batched.as_ptr() + (k_src_offset * std::mem::size_of::<f32>()) as u64;
            let k_dst_ptr = k_cache.as_ptr() + (k_dst_offset * std::mem::size_of::<f32>()) as u64;

            let mut k_src = k_src_ptr;
            let mut k_dst = k_dst_ptr;
            let mut pos_val = pos as u32;
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;

            let scatter_module = self.modules.get_mut(&scatter_key).expect("module exists");
            // SAFETY: Kernel launch with valid pointers - k_src/k_dst from GPU buffers,
            // pos/head_dim/max_len are stack values with stable addresses during call
            unsafe {
                self.stream.launch_kernel(
                    scatter_module,
                    scatter_name,
                    &scatter_config,
                    &mut [
                        std::ptr::from_mut(&mut k_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut pos_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }

            let v_src_offset = seq_idx * kv_dim;
            let v_dst_offset = seq_idx * stride;
            let v_src_ptr = v_batched.as_ptr() + (v_src_offset * std::mem::size_of::<f32>()) as u64;
            let v_dst_ptr = v_cache.as_ptr() + (v_dst_offset * std::mem::size_of::<f32>()) as u64;

            let mut v_src = v_src_ptr;
            let mut v_dst = v_dst_ptr;

            let scatter_module = self.modules.get_mut(&scatter_key).expect("module exists");
            // SAFETY: Kernel launch with valid pointers - v_src/v_dst from GPU buffers,
            // pos/head_dim/max_len are stack values with stable addresses during call
            unsafe {
                self.stream.launch_kernel(
                    scatter_module,
                    scatter_name,
                    &scatter_config,
                    &mut [
                        std::ptr::from_mut(&mut v_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut pos_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Update cache lengths
        for seq_idx in 0..m {
            let pos = positions[seq_idx] as usize;
            if seq_idx < self.batched_kv_lengths.len() {
                self.batched_kv_lengths[seq_idx] = pos + 1;
            }
        }

        // Step 2: Build pointer arrays
        let k_cache_base = k_cache.as_ptr();
        let v_cache_base = v_cache.as_ptr();
        let stride_bytes = (stride * std::mem::size_of::<f32>()) as u64;

        // PMAT-073: Pad pointer/length arrays to match GPU buffer allocation.
        // GPU buffers (batched_k_ptrs, etc.) are allocated for max_kv_slots,
        // but active slots may be fewer (m < max_kv_slots during mid-batch joins).
        // Kernel grid uses m as batch dim, so padding values are never read.
        let buf_len = self.batched_kv_lengths.len();
        let mut k_ptrs: Vec<u64> = (0..m)
            .map(|seq_idx| k_cache_base + seq_idx as u64 * stride_bytes)
            .collect();
        k_ptrs.resize(buf_len, k_cache_base);
        let mut v_ptrs: Vec<u64> = (0..m)
            .map(|seq_idx| v_cache_base + seq_idx as u64 * stride_bytes)
            .collect();
        v_ptrs.resize(buf_len, v_cache_base);
        let mut seq_lens: Vec<u32> = (0..m)
            .map(|seq_idx| {
                // PMAT-076: Zero seq_lens for done slots — flash decode early-exits.
                if seq_idx < self.batched_done_mask.len() && self.batched_done_mask[seq_idx] {
                    0
                } else {
                    self.batched_kv_lengths.get(seq_idx).copied().unwrap_or(1) as u32
                }
            })
            .collect();
        seq_lens.resize(buf_len, 0);

        // Step 3: Compute max chunks needed
        let max_seq_len_actual = seq_lens.iter().copied().max().unwrap_or(1) as usize;
        let max_chunks = (max_seq_len_actual + FLASH_DECODE_CHUNK_SIZE as usize - 1)
            / FLASH_DECODE_CHUNK_SIZE as usize;

        // Step 4: Launch Flash Decoding chunk kernel
        // Compile module BEFORE taking mutable borrows on ptr buffers to
        // avoid borrow-checker conflict (compile_ptx borrows &self).
        let chunk_kernel = FlashDecodingChunkKernel::new(
            max_len as u32,
            head_dim as u32,
            num_heads as u32,
            num_kv_heads as u32,
            m as u32,
        );
        let chunk_kernel_name = chunk_kernel.name();
        let chunk_module_key = format!(
            "flash_decode_chunk_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads
        );

        if !self.modules.contains_key(&chunk_module_key) {
            let chunk_ptx = chunk_kernel.emit_ptx_for_target(&self.kernels.sm_target);
            let module = self.compile_ptx(&chunk_ptx)?;
            self.modules.insert(chunk_module_key.clone(), module);
        }

        let k_ptrs_buf = self.batched_k_ptrs.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-118: batched_k_ptrs not allocated".to_string())
        })?;
        let v_ptrs_buf = self.batched_v_ptrs.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-118: batched_v_ptrs not allocated".to_string())
        })?;
        let seq_lens_buf = self.batched_seq_lens_gpu.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-118: batched_seq_lens_gpu not allocated".to_string())
        })?;

        // CORRECTNESS-013: Use async H2D on self.stream for same-stream ordering.
        // copy_from_host uses stream 0 which has no sync with non-blocking streams.
        // PMAT-088: Buffers may be over-sized from high-water-mark allocation.
        // Create exact-sized views via from_raw_parts for copy_from_host_async.
        let m = seq_lens.len();
        unsafe {
            let mut k_view = GpuBuffer::<u64>::from_raw_parts(k_ptrs_buf.as_ptr(), m);
            let mut v_view = GpuBuffer::<u64>::from_raw_parts(v_ptrs_buf.as_ptr(), m);
            let mut s_view = GpuBuffer::<u32>::from_raw_parts(seq_lens_buf.as_ptr(), m);
            k_view.copy_from_host_async(&k_ptrs, &self.stream)?;
            v_view.copy_from_host_async(&v_ptrs, &self.stream)?;
            s_view.copy_from_host_async(&seq_lens, &self.stream)?;
            std::mem::forget(k_view);
            std::mem::forget(v_view);
            std::mem::forget(s_view);
        }

        let partials_buf = self.flash_decode_partials.as_ref().ok_or_else(|| {
            GpuError::InvalidLaunchConfig(
                "PAR-118: flash_decode_partials not allocated".to_string(),
            )
        })?;

        // Grid: (num_heads, batch_size, max_chunks)
        let chunk_config = LaunchConfig {
            grid: (num_heads as u32, m as u32, max_chunks as u32),
            block: (32, 1, 1),
            shared_mem: 0,
        };

        let mut q_ptr = q_batched.as_ptr();
        let mut k_ptrs_ptr = k_ptrs_buf.as_ptr();
        let mut v_ptrs_ptr = v_ptrs_buf.as_ptr();
        let mut partials_ptr = partials_buf.as_ptr();
        let mut seq_lens_ptr = seq_lens_buf.as_ptr();
        let mut max_chunks_val = max_chunks as u32;

        let chunk_module = self
            .modules
            .get_mut(&chunk_module_key)
            .expect("module just inserted");

        // SAFETY: Kernel launch with valid pointers - all GPU buffer pointers derived from
        // allocated GpuBuffers, max_chunks is stack value with stable address during call
        unsafe {
            self.stream.launch_kernel(
                chunk_module,
                chunk_kernel_name,
                &chunk_config,
                &mut [
                    std::ptr::from_mut(&mut q_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_ptrs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut v_ptrs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut partials_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_lens_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut max_chunks_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Step 5: Launch Flash Decoding reduce kernel
        let reduce_kernel =
            FlashDecodingReduceKernel::new(head_dim as u32, num_heads as u32, m as u32);
        let reduce_kernel_name = reduce_kernel.name();
        let reduce_module_key = format!("flash_decode_reduce_{}_{}", head_dim, num_heads);

        if !self.modules.contains_key(&reduce_module_key) {
            let reduce_ptx = reduce_kernel.emit_ptx_for_target(&self.kernels.sm_target);
            let module = self.compile_ptx(&reduce_ptx)?;
            self.modules.insert(reduce_module_key.clone(), module);
        }

        // Grid: (num_heads, batch_size, 1)
        let reduce_config = LaunchConfig {
            grid: (num_heads as u32, m as u32, 1),
            block: (32, 1, 1),
            shared_mem: 0,
        };

        let mut out_ptr = out_batched.as_ptr();

        let reduce_module = self
            .modules
            .get_mut(&reduce_module_key)
            .expect("module just inserted");

        // SAFETY: Kernel launch with valid pointers - partials/out/seq_lens from GPU buffers,
        // max_chunks is stack value with stable address during call
        unsafe {
            self.stream.launch_kernel(
                reduce_module,
                reduce_kernel_name,
                &reduce_config,
                &mut [
                    std::ptr::from_mut(&mut partials_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut out_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_lens_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut max_chunks_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// Tensor Core attention using WMMA for FP16 matrix operations (PARITY-001.3)
    ///
    /// Uses FP16 Tensor Cores (WMMA) for Q×K^T and attention×V computation.
    /// Expected 4-10x speedup over FP32 FlashAttention on Tensor Core GPUs.
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor [n_heads, seq_len, head_dim] as FP32 (converted to FP16)
    /// * `k` - Key tensor [n_heads, seq_len, head_dim] as FP32 (converted to FP16)
    /// * `v` - Value tensor [n_heads, seq_len, head_dim] as FP32 (converted to FP16)
    /// * `output` - Output tensor [n_heads, seq_len, head_dim] (FP32 accumulator)
    /// * `seq_len` - Sequence length (must be multiple of 16 for WMMA)
    /// * `head_dim` - Dimension per head (must be multiple of 16 for WMMA)
    /// * `n_heads` - Number of attention heads
    /// * `causal` - Whether to apply causal masking
    ///
    /// # Performance
    ///
    /// RTX 4090: 330 TFLOPS FP16 vs 83 TFLOPS FP32 (4x theoretical speedup)
    /// Target: <2ms per token vs 79ms FP32 baseline (~40x actual speedup)
    #[allow(clippy::too_many_arguments)]
    pub fn tensor_core_attention(
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
        // WMMA requires dimensions to be multiples of 16
        if !seq_len.is_multiple_of(16) || !head_dim.is_multiple_of(16) {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "Tensor Core attention requires dimensions multiple of 16: seq_len={}, head_dim={}",
                seq_len, head_dim
            )));
        }

        let head_size = (seq_len * head_dim) as usize;
        let total_size = head_size * n_heads as usize;

        // Validate input sizes
        if q.len() != total_size
            || k.len() != total_size
            || v.len() != total_size
            || output.len() != total_size
        {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "Tensor Core attention size mismatch: expected {} ({}×{}×{}), got Q[{}] K[{}] V[{}] O[{}]",
                total_size, n_heads, seq_len, head_dim,
                q.len(), k.len(), v.len(), output.len()
            )));
        }

        // Track memory allocation (FP32 buffers - conversion happens on GPU)
        self.memory_pool.record_allocation(total_size * 4 * 4);

        // Generate Tensor Core attention kernel
        let kernel_type = KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!(
            "tensor_core_attn_{}_{}_{}_{}",
            seq_len, head_dim, n_heads, causal
        );

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            #[cfg(test)]
            eprintln!("Generated Tensor Core attention PTX:\n{}", ptx);
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

        // Launch configuration for Tensor Core attention:
        // Grid.x = ceil(seq_len / 16) - number of 16×16 WMMA tiles
        // Grid.y = n_heads
        // Threads = 256 (8 warps per block for WMMA)
        let num_tiles = (seq_len + 15) / 16;
        let config = LaunchConfig::grid_2d(num_tiles, n_heads, 256, 1);

        // Get raw pointers for kernel args
        let mut ptr_q = buf_q.as_ptr();
        let mut ptr_k = buf_k.as_ptr();
        let mut ptr_v = buf_v.as_ptr();
        let mut ptr_output = buf_output.as_ptr();
        let mut seq_len_val = seq_len;
        let mut head_dim_val = head_dim;
        let mut n_heads_val = n_heads;

        // Launch kernel
        // SAFETY: Buffers are valid, dimensions validated
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
