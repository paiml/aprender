impl CudaExecutor {

    /// PAR-118: Graph-compatible Flash Decoding for single-sequence decode (M=1).
    ///
    /// Uses pre-allocated per-layer K/V pointer buffers and static max_chunks grid
    /// dimensions for CUDA graph compatibility. seq_len is read from either
    /// `seq_len_buf` (graph mode) or passed directly.
    ///
    /// Five-Whys: attention uses only 28 blocks (one per head) on RTX 4090 (128 SMs),
    /// leaving 78% of SMs idle. Flash Decoding splits KV scan into max_chunks blocks
    /// per head, giving 28 × max_chunks = 224 blocks for 8 chunks. This 8x increase
    /// in block count directly improves SM occupancy and memory-level parallelism.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_decoding_graphed(
        &mut self,
        layer_idx: usize,
        q_gpu: &GpuBuffer<f32>,
        out_gpu: &GpuBuffer<f32>,
        _use_graph_mode: bool,
        seq_len: u32,
    ) -> Result<(), GpuError> {
        use trueno_gpu::kernels::{FlashDecodingChunkKernel, FlashDecodingReduceKernel, Kernel};

        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let max_chunks = self.flash_decode_max_chunks;

        // Get pre-allocated per-layer K/V pointer buffers
        let k_ptrs_buf = self.flash_decode_k_ptrs.get(&layer_idx).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "PAR-118: Flash Decoding K pointer buffer not found for layer {layer_idx}"
            ))
        })?;
        let v_ptrs_buf = self.flash_decode_v_ptrs.get(&layer_idx).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!(
                "PAR-118: Flash Decoding V pointer buffer not found for layer {layer_idx}"
            ))
        })?;
        let partials_buf = self.flash_decode_partials.as_ref().ok_or_else(|| {
            GpuError::InvalidLaunchConfig(
                "PAR-118: Flash Decoding partials buffer not allocated".to_string(),
            )
        })?;

        // Compile chunk kernel
        let chunk_kernel = FlashDecodingChunkKernel::new(
            max_len as u32,
            head_dim as u32,
            num_heads as u32,
            num_kv_heads as u32,
            1, // M=1 for single-sequence decode
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

        // PAR-118: Static grid dimensions for CUDA graph compatibility.
        // max_chunks is fixed based on max_seq_len, not actual seq_len.
        // The chunk kernel checks actual seq_len and early-exits for empty chunks
        // (stores sentinel max=-inf, sum=0 which the reduce kernel skips).
        let chunk_config = LaunchConfig {
            grid: (num_heads as u32, 1, max_chunks as u32),
            block: (32, 1, 1),
            shared_mem: 0,
        };

        let mut q_ptr = q_gpu.as_ptr();
        let mut k_ptrs_ptr = k_ptrs_buf.as_ptr();
        let mut v_ptrs_ptr = v_ptrs_buf.as_ptr();
        let mut partials_ptr = partials_buf.as_ptr();
        let mut max_chunks_val = max_chunks as u32;

        // PAR-118: seq_len pointer selection depends on capture vs eager mode.
        let mut seq_lens_ptr = if self.is_capturing {
            // During graph capture: use seq_len_buf (already populated by prepare_capture_buffers).
            // No copy_from_host — forbidden during capture. During graph replay, seq_len_buf
            // is updated via async copy before graph launch, so captured kernels read fresh values.
            self.seq_len_buf.as_ref().ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "PAR-118: seq_len_buf not initialized for graph capture".to_string(),
                )
            })?.as_ptr()
        } else {
            // Eager path: use dedicated flash_decode_seq_lens_buf with explicit copy.
            // CRITICAL: Do NOT use seq_len_buf here — may have stale values after graph
            // capture failure (error 901 fallback).
            let buf = self.flash_decode_seq_lens_buf.as_mut().ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "PAR-118: flash_decode_seq_lens_buf not initialized".to_string(),
                )
            })?;
            buf.copy_from_host(&[seq_len])?;
            buf.as_ptr()
        };

        {
            let chunk_module = self
                .modules
                .get_mut(&chunk_module_key)
                .expect("module just compiled");

            // SAFETY: All pointers from pre-allocated GPU buffers with stable addresses.
            // seq_lens_ptr points to a long-lived buffer that outlives the kernel execution.
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
        }

        // trueno#243: Record chunk kernel for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&chunk_module_key).expect("module exists");
            let func = module.get_function(chunk_kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config: chunk_config,
                arg_data: vec![q_ptr, k_ptrs_ptr, v_ptrs_ptr, partials_ptr, seq_lens_ptr, max_chunks_val as u64],
            });
        }

        // PAR-118-FIX: No synchronize needed between chunk and reduce kernels.
        // CUDA stream semantics guarantee all blocks from kernel A complete before
        // kernel B starts on the same stream. The previous sync was a CPU-GPU
        // round-trip (~10µs) × 28 layers = ~280µs/token of pure overhead.

        // Reduce kernel: one block per head, reduces across chunks
        let reduce_kernel = FlashDecodingReduceKernel::new(
            head_dim as u32,
            num_heads as u32,
            1, // M=1
        );
        let reduce_kernel_name = reduce_kernel.name();
        let reduce_module_key = format!("flash_decode_reduce_{}_{}", head_dim, num_heads);

        if !self.modules.contains_key(&reduce_module_key) {
            let reduce_ptx = reduce_kernel.emit_ptx_for_target(&self.kernels.sm_target);
            let module = self.compile_ptx(&reduce_ptx)?;
            self.modules.insert(reduce_module_key.clone(), module);
        }

        let reduce_config = LaunchConfig {
            grid: (num_heads as u32, 1, 1),
            block: (32, 1, 1),
            shared_mem: 0,
        };

        let mut out_ptr = out_gpu.as_ptr();
        let mut partials_ptr2 = partials_buf.as_ptr();
        let mut max_chunks_val2 = max_chunks as u32;

        // Reuse same seq_lens_ptr from chunk kernel (already set above, persistent buffer)
        {
            let reduce_module = self
                .modules
                .get_mut(&reduce_module_key)
                .expect("module just compiled");

            // SAFETY: All pointers from pre-allocated GPU buffers with stable addresses.
            // seq_lens_ptr uses the same long-lived buffer as the chunk kernel.
            unsafe {
                self.stream.launch_kernel(
                    reduce_module,
                    reduce_kernel_name,
                    &reduce_config,
                    &mut [
                        std::ptr::from_mut(&mut partials_ptr2) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut out_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut seq_lens_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_chunks_val2) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // trueno#243: Record reduce kernel for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&reduce_module_key).expect("module exists");
            let func = module.get_function(reduce_kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config: reduce_config,
                arg_data: vec![partials_ptr2, out_ptr, seq_lens_ptr, max_chunks_val2 as u64],
            });
        }

        Ok(())
    }
}
