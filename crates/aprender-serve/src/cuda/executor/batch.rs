impl CudaExecutor {

    /// PAR-119: Batched incremental attention for M sequences in parallel
    ///
    /// This eliminates the sequential attention bottleneck by processing all M sequences
    /// in a single kernel launch. Uses BatchedIncrementalAttentionKernel with pointer arrays
    /// to access per-sequence KV caches.
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
    pub fn batched_incremental_attention_into(
        &mut self,
        layer_idx: usize,
        q_batched: &GpuBuffer<f32>,
        k_batched: &GpuBuffer<f32>,
        v_batched: &GpuBuffer<f32>,
        out_batched: &GpuBuffer<f32>,
        m: usize,
        positions: &[u32],
    ) -> Result<(), GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let stride = self.batched_kv_stride;

        if stride == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-119: Batched KV cache not initialized (call init_batched_kv_cache_gpu first)"
                    .to_string(),
            ));
        }

        // Get batched KV cache buffers for this layer
        let k_cache = self.batched_kv_k_caches.get(&layer_idx).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "PAR-119: Batched K cache not found for layer {}",
                layer_idx
            ))
        })?;
        let v_cache = self.batched_kv_v_caches.get(&layer_idx).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "PAR-119: Batched V cache not found for layer {}",
                layer_idx
            ))
        })?;

        // Step 1: Scatter K/V to per-sequence caches
        // For each sequence seq_idx, scatter to cache[seq_idx * stride + pos * head_dim * num_kv_heads]
        let kv_dim = num_kv_heads * head_dim;
        let scatter_config = LaunchConfig {
            grid: (num_kv_heads as u32, 1, 1),
            block: (head_dim as u32, 1, 1),
            shared_mem: 0,
        };

        // Get or compile scatter kernel
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

        // PMAT-046: Batched KV scatter — 2 launches (K+V) regardless of M.
        // Previous: 2×M launches per layer (was 224 at M=4, L=28).
        // Now: 2 launches per layer (56 at L=28). Saves 168 launches/step.
        //
        // Kernel layout: Grid (num_kv_heads, M, 1), Block (head_dim, 1, 1)
        //   seq_idx = blockIdx.y
        //   head_idx = blockIdx.x
        //   elem_idx = threadIdx.x
        //   src: src_base[seq_idx * kv_dim + head_idx * head_dim + elem_idx]
        //   dst: dst_base[seq_idx * stride + (head_idx * max_len + positions[seq_idx]) * head_dim + elem_idx]
        let batched_scatter_key = format!("batched_kv_scatter_{}_{}_{}", num_kv_heads, head_dim, max_len);
        if !self.modules.contains_key(&batched_scatter_key) {
            let ptx = format!(
                r#".version 7.0
.target sm_70
.address_size 64

.visible .entry batched_kv_cache_scatter(
    .param .u64 src_base,
    .param .u64 dst_base,
    .param .u64 positions_ptr,
    .param .u32 stride_param,
    .param .u32 kv_dim_param
) {{
    .reg .u64 %rd<16>;
    .reg .u32 %r<16>;
    .reg .f32 %f<2>;
    .reg .pred %p;

    // elem_idx = threadIdx.x
    mov.u32 %r0, %tid.x;
    // head_idx = blockIdx.x
    mov.u32 %r1, %ctaid.x;
    // seq_idx = blockIdx.y
    mov.u32 %r2, %ctaid.y;

    // bounds check: elem_idx < head_dim
    setp.ge.u32 %p, %r0, {head_dim};
    @%p bra DONE;

    // Load positions[seq_idx]
    ld.param.u64 %rd0, [positions_ptr];
    mul.wide.u32 %rd1, %r2, 4;          // seq_idx * 4
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r3, [%rd2];          // pos = positions[seq_idx]

    // Source: src_base + (seq_idx * kv_dim + head_idx * head_dim + elem_idx) * 4
    ld.param.u32 %r4, [kv_dim_param];   // kv_dim
    mul.lo.u32 %r5, %r2, %r4;           // seq_idx * kv_dim
    mul.lo.u32 %r6, %r1, {head_dim};    // head_idx * head_dim
    add.u32 %r5, %r5, %r6;              // + head_idx * head_dim
    add.u32 %r5, %r5, %r0;              // + elem_idx
    mul.wide.u32 %rd3, %r5, 4;
    ld.param.u64 %rd4, [src_base];
    add.u64 %rd5, %rd4, %rd3;           // src_addr

    // Dest: dst_base + (seq_idx * stride + (head_idx * max_len + pos) * head_dim + elem_idx) * 4
    ld.param.u32 %r7, [stride_param];   // stride
    mul.lo.u32 %r8, %r2, %r7;           // seq_idx * stride
    mul.lo.u32 %r9, %r1, {max_len};     // head_idx * max_len
    add.u32 %r9, %r9, %r3;              // + pos
    mul.lo.u32 %r9, %r9, {head_dim};    // * head_dim
    add.u32 %r8, %r8, %r9;              // + (head_idx * max_len + pos) * head_dim
    add.u32 %r8, %r8, %r0;              // + elem_idx
    mul.wide.u32 %rd6, %r8, 4;
    ld.param.u64 %rd7, [dst_base];
    add.u64 %rd8, %rd7, %rd6;           // dst_addr

    // Copy: dst[...] = src[...]
    ld.global.f32 %f0, [%rd5];
    st.global.f32 [%rd8], %f0;

DONE:
    ret;
}}"#,
                head_dim = head_dim,
                max_len = max_len,
            );
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(batched_scatter_key.clone(), module);
        }

        // Upload positions to GPU
        let positions_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let positions_buf = self
            .workspace
            .positions_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-046: positions_buf not initialized".to_string())
            })?;
        let positions_buf_ptr = positions_buf.as_ptr();
        // SAFETY: positions_buf is valid workspace allocation sized for M positions
        let mut positions_gpu = unsafe { GpuBuffer::<u32>::from_raw_parts(positions_buf_ptr, m) };
        if !self.is_capturing {
            positions_gpu.copy_from_host(&positions_u32)?;
        }

        let batched_scatter_config = LaunchConfig {
            grid: (num_kv_heads as u32, m as u32, 1),
            block: (head_dim as u32, 1, 1),
            shared_mem: 0,
        };

        let batched_scatter_module = self.modules.get_mut(&batched_scatter_key).expect("module exists");
        let mut k_src_base = k_batched.as_ptr();
        let mut k_dst_base = k_cache.as_ptr();
        let mut pos_ptr = positions_buf_ptr;
        let mut stride_val = stride as u32;
        let mut kv_dim_val = kv_dim as u32;

        // CORRECTNESS-012b: Use self.stream for ALL scatter/attention in batched path.
        // Five-Whys: 6.7% of c=4 requests produce 0 output tokens.
        // Why? First decode token is EOS. Why? KV cache contains garbage.
        // Why? Scatter reads stale K/V from QKV projection. Why? QKV runs on
        // self.stream, scatter ran on compute_stream with no inter-stream sync.
        // Root cause: Same as CORRECTNESS-012 (kv_scatter.rs) but unfixed in batch path.
        // Fix: Use self.stream for scatter/attention, matching the M=1 fix.
        // SAFETY: src/dst are valid GPU allocs, positions_buf populated above, stride/kv_dim bounds verified
        unsafe {
            self.stream.launch_kernel(
                batched_scatter_module,
                "batched_kv_cache_scatter",
                &batched_scatter_config,
                &mut [
                    std::ptr::from_mut(&mut k_src_base) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_dst_base) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut pos_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut stride_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut kv_dim_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Scatter V (same kernel, different src/dst)
        let batched_scatter_module = self.modules.get_mut(&batched_scatter_key).expect("module exists");
        let mut v_src_base = v_batched.as_ptr();
        let mut v_dst_base = v_cache.as_ptr();
        let mut pos_ptr2 = positions_buf_ptr;
        let mut stride_val2 = stride as u32;
        let mut kv_dim_val2 = kv_dim as u32;

        // CORRECTNESS-012b: Same stream as K scatter (self.stream, not compute_stream)
        // SAFETY: v_batched/v_cache are valid GPU allocs, positions_buf populated above
        unsafe {
            self.stream.launch_kernel(
                batched_scatter_module,
                "batched_kv_cache_scatter",
                &batched_scatter_config,
                &mut [
                    std::ptr::from_mut(&mut v_src_base) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut v_dst_base) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut pos_ptr2) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut stride_val2) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut kv_dim_val2) as *mut std::ffi::c_void,
                ],
            )?;
        }

        std::mem::forget(positions_gpu);

        // PMAT-045: Skip CPU-side state update during CUDA graph capture.
        // During capture, kernels are recorded, not executed — data values don't matter.
        // On replay, batched_kv_lengths is updated by the non-captured caller path.
        if !self.is_capturing {
            for seq_idx in 0..m {
                let pos = positions[seq_idx] as usize;
                if seq_idx < self.batched_kv_lengths.len() {
                    self.batched_kv_lengths[seq_idx] = pos + 1;
                }
            }
        }

        // Step 2: Build pointer arrays for batched attention
        // Each pointer points to the start of that sequence's KV cache
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
                // PMAT-076: Zero seq_lens for done slots — attention kernel early-exits
                // (loop condition: pos < seq_len, so seq_len=0 → zero KV iterations).
                if seq_idx < self.batched_done_mask.len() && self.batched_done_mask[seq_idx] {
                    0
                } else {
                    self.batched_kv_lengths.get(seq_idx).copied().unwrap_or(1) as u32
                }
            })
            .collect();
        seq_lens.resize(buf_len, 0);

        // Step 3: Launch batched attention kernel
        // Compile module BEFORE taking mutable borrows on ptr buffers to
        // avoid borrow-checker conflict (compile_ptx borrows &self).
        let kernel = BatchedIncrementalAttentionKernel::new(
            max_len as u32,
            head_dim as u32,
            num_heads as u32,
            num_kv_heads as u32,
            m as u32,
        );
        let kernel_name = kernel.name();
        let module_key = format!(
            "batched_incr_attn_{}_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads, m
        );

        if !self.modules.contains_key(&module_key) {
            // PAR-119: Use emit_ptx() to get full module with version/target headers
            let ptx_source = kernel.emit_ptx_for_target(&self.kernels.sm_target);
            let module = self.compile_ptx(&ptx_source)?;
            self.modules.insert(module_key.clone(), module);
        }

        let k_ptrs_buf = self.batched_k_ptrs.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-119: batched_k_ptrs not allocated".to_string())
        })?;
        let v_ptrs_buf = self.batched_v_ptrs.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-119: batched_v_ptrs not allocated".to_string())
        })?;
        let seq_lens_buf = self.batched_seq_lens_gpu.as_mut().ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-119: batched_seq_lens_gpu not allocated".to_string())
        })?;

        // PMAT-045: Skip copy_from_host during CUDA graph capture.
        // cuMemcpyHtoD is not capturable — causes CUDA_ERROR_ILLEGAL_ADDRESS.
        // GH-141: During capture, use per-layer pointer buffers instead of shared
        // buffer — each layer's KV cache is at a different address, and the graph
        // records the pointer passed to the kernel. Per-layer buffers are static
        // (pre-populated in init_batched_kv_cache_gpu).
        if !self.is_capturing {
            // PMAT-088: Use copy_from_host_at(0) — buffers may be over-sized from
            // high-water-mark allocation (e.g., allocated for M=4 but used at M=3).
            // copy_from_host requires exact length; copy_from_host_at allows partial.
            k_ptrs_buf.copy_from_host_at(&k_ptrs, 0).map_err(|e| {
                GpuError::Transfer(format!(
                    "PMAT-088c k_ptrs: host={} device={}: {e}",
                    k_ptrs.len(), k_ptrs_buf.len(),
                ))
            })?;
            v_ptrs_buf.copy_from_host_at(&v_ptrs, 0).map_err(|e| {
                GpuError::Transfer(format!(
                    "PMAT-088c v_ptrs: host={} device={}: {e}",
                    v_ptrs.len(), v_ptrs_buf.len(),
                ))
            })?;
            seq_lens_buf.copy_from_host_at(&seq_lens, 0).map_err(|e| {
                GpuError::Transfer(format!(
                    "PMAT-088c seq_lens: host={} device={}: {e}",
                    seq_lens.len(), seq_lens_buf.len(),
                ))
            })?;
        }

        // GH-141: Choose pointer source — per-layer (capture) or shared (non-capture)
        let (k_ptrs_ptr_val, v_ptrs_ptr_val) = if self.is_capturing {
            // Per-layer buffers: graph records correct per-layer addresses
            let k_ptr = self.batched_k_ptrs_per_layer.get(&layer_idx)
                .map_or(k_ptrs_buf.as_ptr(), |b| b.as_ptr());
            let v_ptr = self.batched_v_ptrs_per_layer.get(&layer_idx)
                .map_or(v_ptrs_buf.as_ptr(), |b| b.as_ptr());
            (k_ptr, v_ptr)
        } else {
            // Shared buffers: updated per layer via copy_from_host above
            (k_ptrs_buf.as_ptr(), v_ptrs_buf.as_ptr())
        };

        let module = self
            .modules
            .get_mut(&module_key)
            .expect("module just inserted");

        // Grid: (num_heads, batch_size, 1), Block: (32, 1, 1)
        let config = LaunchConfig {
            grid: (num_heads as u32, m as u32, 1),
            block: (32, 1, 1),
            shared_mem: 0,
        };

        let mut q_ptr = q_batched.as_ptr();
        let mut k_ptrs_ptr = k_ptrs_ptr_val;
        let mut v_ptrs_ptr = v_ptrs_ptr_val;
        let mut out_ptr = out_batched.as_ptr();
        let mut seq_lens_ptr = seq_lens_buf.as_ptr();

        // CORRECTNESS-012b: Use self.stream for attention (same as scatter above)
        // SAFETY: Unsafe operation with validated invariants
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut q_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_ptrs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut v_ptrs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut out_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut seq_lens_ptr) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    // =========================================================================
    // PAR-118: Flash Decoding - Split-K Attention for 2X Ollama Performance
    // =========================================================================

    /// Initialize Flash Decoding buffers for split-K attention.
    ///
    /// Flash Decoding splits the KV cache into chunks processed in parallel,
    /// then reduces partial results. This amortizes memory bandwidth across
    /// multiple thread blocks, achieving higher throughput for long sequences.
    ///
    /// # Arguments
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Head dimension
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `batch_size` - Batch size (M)
    ///
    /// # Performance
    /// - Expected 1.5-2x speedup over sequential attention for seq_len > 128
    /// - Minimal overhead for short sequences (< 128 positions)
    pub fn init_flash_decoding(
        &mut self,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        batch_size: usize,
    ) -> Result<(), GpuError> {
        use trueno_gpu::kernels::FLASH_DECODE_CHUNK_SIZE;

        // Calculate partials buffer size
        // Layout: [M, num_heads, max_chunks, head_dim + 2]
        let max_chunks =
            (max_seq_len + FLASH_DECODE_CHUNK_SIZE as usize - 1) / FLASH_DECODE_CHUNK_SIZE as usize;
        let partials_per_head = max_chunks * (head_dim + 2);
        let total_partials = batch_size * num_heads * partials_per_head;

        // Allocate partials buffer
        self.flash_decode_partials = Some(GpuBuffer::new(&self.context, total_partials)?);
        self.flash_decode_max_seq_len = max_seq_len;
        self.flash_decode_max_chunks = max_chunks;
        self.flash_decode_enabled = true;

        // PAR-118 + realizr#214: Persistent seq_lens buffer — sized for max_batch (not M=1)
        // to support per-batch CUDA graph capture. During M=1 decode, only index 0 is used.
        // During M>1 batched graph replay, all M indices are updated before graph launch.
        let max_batch = std::env::var("CUDA_MAX_BATCH")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(32usize);
        let initial_seq_lens = vec![0u32; max_batch];
        self.flash_decode_seq_lens_buf = Some(GpuBuffer::from_host(&self.context, &initial_seq_lens)?);

        // PAR-118: Populate per-layer K/V pointer buffers for graph-compatible Flash Decoding.
        // These contain the GPU base address of each layer's KV cache, enabling the
        // FlashDecodingChunkKernel to read k_ptrs[0] / v_ptrs[0] indirectly.
        // Addresses are stable because KV cache buffers are allocated once in init_kv_cache_gpu.
        let num_layers = self.kv_cache_lengths.len().max(
            self.kv_cache_gpu
                .keys()
                .filter_map(|k| {
                    k.strip_prefix("kv_")
                        .and_then(|s| s.strip_suffix("_k"))
                        .and_then(|s| s.parse::<usize>().ok())
                })
                .max()
                .map_or(0, |m| m + 1),
        );

        for layer_idx in 0..num_layers {
            let k_key = format!("kv_{layer_idx}_k");
            let v_key = format!("kv_{layer_idx}_v");

            if let (Some(k_cache), Some(v_cache)) =
                (self.kv_cache_gpu.get(&k_key), self.kv_cache_gpu.get(&v_key))
            {
                // realizr#214: Size for max_batch to support per-batch graph capture.
                // All M slots point to the same cache base; the kernel indexes via batch_idx.
                let k_ptrs = vec![k_cache.as_ptr(); max_batch];
                let v_ptrs = vec![v_cache.as_ptr(); max_batch];
                let k_ptr_buf = GpuBuffer::from_host(&self.context, &k_ptrs)?;
                let v_ptr_buf = GpuBuffer::from_host(&self.context, &v_ptrs)?;
                self.flash_decode_k_ptrs.insert(layer_idx, k_ptr_buf);
                self.flash_decode_v_ptrs.insert(layer_idx, v_ptr_buf);
            }
        }

        // PAR-118: Pre-compile Flash Decoding PTX modules to avoid compilation during
        // CUDA graph capture (which causes CUDA_ERROR_UNKNOWN/901).
        // Must happen BEFORE any graph capture attempt.
        {
            use trueno_gpu::kernels::{
                FlashDecodingChunkKernel, FlashDecodingReduceKernel, Kernel,
            };

            let num_kv_heads = self.kv_num_kv_heads;
            let max_len = self.kv_cache_max_len;

            let chunk_kernel = FlashDecodingChunkKernel::new(
                max_len as u32,
                head_dim as u32,
                num_heads as u32,
                num_kv_heads as u32,
                1, // M=1 for single-sequence decode
            );
            let chunk_module_key = format!(
                "flash_decode_chunk_{}_{}_{}_{}",
                max_len, head_dim, num_heads, num_kv_heads
            );
            if !self.modules.contains_key(&chunk_module_key) {
                let chunk_ptx = chunk_kernel.emit_ptx_for_target(&self.kernels.sm_target);
                let module = self.compile_ptx(&chunk_ptx)?;
                self.modules.insert(chunk_module_key, module);
            }

            let reduce_kernel = FlashDecodingReduceKernel::new(
                head_dim as u32,
                num_heads as u32,
                1, // M=1
            );
            let reduce_module_key = format!("flash_decode_reduce_{}_{}", head_dim, num_heads);
            if !self.modules.contains_key(&reduce_module_key) {
                let reduce_ptx = reduce_kernel.emit_ptx_for_target(&self.kernels.sm_target);
                let module = self.compile_ptx(&reduce_ptx)?;
                self.modules.insert(reduce_module_key, module);
            }
        }

        Ok(())
    }
}
