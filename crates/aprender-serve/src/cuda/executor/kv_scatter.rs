impl CudaExecutor {

    /// PAR-052: Scatter K and V tensors into the KV cache for a given layer.
    ///
    /// Dispatches either indirect scatter (for graph capture mode, when `position_buf`
    /// is present) or direct scatter (normal mode) kernels. Each path launches two
    /// kernels: one for K, one for V.
    ///
    /// # Arguments
    /// * `layer_idx` - Transformer layer index
    /// * `k_gpu` - Key tensor on GPU
    /// * `v_gpu` - Value tensor on GPU
    /// * `cache_len` - Current cache length before this token
    /// * `use_stateless` - CORRECTNESS-013: stateless mode writes to position 0
    fn scatter_kv_to_cache(
        &mut self,
        layer_idx: usize,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
        cache_len: usize,
        use_stateless: bool,
    ) -> Result<(), GpuError> {
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;

        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);

        // CORRECTNESS-001 FIX: Launch config must match kernel expectations:
        // - Each block handles one KV head (head_idx = ctaid.x)
        // - Each thread handles one element (elem_idx = tid.x)
        // Grid: num_kv_heads blocks, Block: head_dim threads
        let config = LaunchConfig {
            grid: (num_kv_heads as u32, 1, 1),
            block: (head_dim as u32, 1, 1),
            shared_mem: 0,
        };

        // PAR-061: Use indirect scatter during graph capture to avoid baking position
        // PAR-069: Use graph mode (indirect scatter) ONLY when position_buf is initialized
        if let Some(ref pos_buf) = self.position_buf {
            self.scatter_kv_indirect(layer_idx, k_gpu, v_gpu, &k_key, &v_key, &config, pos_buf.as_ptr())?;
        } else {
            // PAR-069: Normal mode (no graph capture) - use direct scatter kernel
            // CORRECTNESS-013: In stateless mode, always write to position 0
            let position_val = if use_stateless {
                0u32
            } else {
                cache_len as u32
            };
            self.scatter_kv_direct(layer_idx, k_gpu, v_gpu, &k_key, &v_key, &config, position_val)?;
        }

        Ok(())
    }

    /// PAR-061: Indirect scatter path for graph capture mode.
    /// Reads position from a device buffer pointer instead of a host value.
    fn scatter_kv_indirect(
        &mut self,
        layer_idx: usize,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
        k_key: &str,
        v_key: &str,
        config: &LaunchConfig,
        pos_buf_ptr: u64,
    ) -> Result<(), GpuError> {
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;

        let scatter_type = KernelType::KvCacheScatterIndirect {
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            max_len: max_len as u32,
        };
        let scatter_name = self.kernels.kernel_name(&scatter_type);
        let scatter_ptx = self.kernels.generate_ptx(&scatter_type);
        let scatter_key = format!("kv_scatter_indirect_{}_{}", num_kv_heads, head_dim);

        if !self.modules.contains_key(&scatter_key) {
            let module = self.compile_ptx(&scatter_ptx)?;
            self.modules.insert(scatter_key.clone(), module);
        }

        // Scatter K
        let k_record_args;
        {
            let k_buf = self.kv_cache_gpu.get_mut(k_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-052: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            let mut k_src_ptr = k_gpu.as_ptr();
            let mut k_dst_ptr = k_buf.as_ptr();
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;
            let mut pos_ptr = pos_buf_ptr;

            // trueno#243: Capture args for recording before module borrow
            k_record_args = vec![k_src_ptr, k_dst_ptr, pos_ptr, head_dim_val as u64, max_len_val as u64];

            let scatter_module = self.modules.get_mut(&scatter_key).expect("just inserted");

            // CORRECTNESS-001 FIX: Kernel expects (src, cache, pos_ptr, head_dim, max_len)
            // CORRECTNESS-011: Use self.stream for graph capture (graph captures on stream, not compute_stream)
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    scatter_module,
                    scatter_name,
                    config,
                    &mut [
                        std::ptr::from_mut(&mut k_src_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_dst_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut pos_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // trueno#243: Record K scatter for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&scatter_key).expect("module exists");
            let func = module.get_function(scatter_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config: config.clone(),
                arg_data: k_record_args,
            });
        }

        // Scatter V
        let v_record_args;
        {
            let scatter_module = self.modules.get_mut(&scatter_key).expect("module exists");
            let v_buf = self.kv_cache_gpu.get_mut(v_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-052: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            let mut v_src_ptr = v_gpu.as_ptr();
            let mut v_dst_ptr = v_buf.as_ptr();
            let mut pos_ptr = pos_buf_ptr;
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;

            // trueno#243: Capture args for recording
            v_record_args = vec![v_src_ptr, v_dst_ptr, pos_ptr, head_dim_val as u64, max_len_val as u64];

            // CORRECTNESS-001 FIX: Same fix for V scatter
            // CORRECTNESS-011: Use self.stream for graph capture
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    scatter_module,
                    scatter_name,
                    config,
                    &mut [
                        std::ptr::from_mut(&mut v_src_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_dst_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut pos_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // trueno#243: Record V scatter for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&scatter_key).expect("module exists");
            let func = module.get_function(scatter_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config: config.clone(),
                arg_data: v_record_args,
            });
        }

        Ok(())
    }

    /// PAR-069: Direct scatter path for normal (non-graph-capture) mode.
    /// Passes position as a host u32 value baked into the kernel args.
    fn scatter_kv_direct(
        &mut self,
        layer_idx: usize,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
        k_key: &str,
        v_key: &str,
        config: &LaunchConfig,
        position_val: u32,
    ) -> Result<(), GpuError> {
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;

        let scatter_type = KernelType::KvCacheScatter {
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            max_len: max_len as u32,
        };
        let scatter_name = self.kernels.kernel_name(&scatter_type);
        let scatter_ptx = self.kernels.generate_ptx(&scatter_type);
        let scatter_key = format!("kv_scatter_{}_{}", num_kv_heads, head_dim);

        if !self.modules.contains_key(&scatter_key) {
            let module = self.compile_ptx(&scatter_ptx)?;
            self.modules.insert(scatter_key.clone(), module);
        }

        // Scatter K
        {
            let k_buf = self.kv_cache_gpu.get_mut(k_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-052: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            let mut k_src_ptr = k_gpu.as_ptr();
            let mut k_dst_ptr = k_buf.as_ptr();
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;
            let mut position_val = position_val;

            let scatter_module = self.modules.get_mut(&scatter_key).expect("just inserted");

            // CORRECTNESS-001 FIX: Kernel expects (src, cache, pos, head_dim, max_len)
            // Fixed parameter order: pos is 3rd, removed extra num_heads_val
            // CORRECTNESS-012: Use self.stream to match attention kernel stream
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    scatter_module,
                    scatter_name,
                    config,
                    &mut [
                        std::ptr::from_mut(&mut k_src_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_dst_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut position_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Scatter V
        {
            let scatter_module = self.modules.get_mut(&scatter_key).expect("module exists");
            let v_buf = self.kv_cache_gpu.get_mut(v_key).ok_or_else(|| {
                GpuError::InvalidLaunchConfig(format!(
                    "PAR-052: KV cache not initialized for layer {}",
                    layer_idx
                ))
            })?;
            let mut v_src_ptr = v_gpu.as_ptr();
            let mut v_dst_ptr = v_buf.as_ptr();
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;
            let mut position_val = position_val;

            // CORRECTNESS-001 FIX: Same fix for V scatter
            // CORRECTNESS-012: Use self.stream to match attention kernel stream
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    scatter_module,
                    scatter_name,
                    config,
                    &mut [
                        std::ptr::from_mut(&mut v_src_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_dst_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut position_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        Ok(())
    }

    /// PAR-074: Select, compile, and launch the attention kernel.
    ///
    /// Handles adaptive kernel selection (single-warp vs multi-warp),
    /// PTX compilation with module caching, and kernel launch with
    /// either indirect (graph capture) or direct seq_len passing.
    fn launch_attention_kernel(
        &mut self,
        layer_idx: usize,
        q_gpu: &GpuBuffer<f32>,
        out_gpu: &GpuBuffer<f32>,
        new_len: usize,
        use_graph_mode: bool,
        skip_debug: bool,
    ) -> Result<(), GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;

        let k_key = format!("kv_{}_k", layer_idx);
        let v_key = format!("kv_{}_v", layer_idx);

        // PAR-074: Adaptive attention kernel selection based on sequence length
        // - Short sequences (< 128): Use single-warp kernel (less overhead, ~1-2us/token)
        // - Long sequences (>= 128): Use multi-warp kernel (parallel processing)
        //
        // Five-Whys Root Cause: Multi-warp has 4x warp synchronization overhead
        // that dominates at short sequences where there's not enough parallelism.
        //
        // CORRECTNESS-009: Single-warp kernel only handles head_dim <= 64 (2 elements/thread)
        // For head_dim > 64 (e.g., Qwen 2.5 with head_dim=128), must use multi-warp kernel
        // which handles 4 elements per thread (q0, q1, q2, q3 at offsets 0, 32, 64, 96)
        let use_single_warp = new_len < 128 && head_dim <= 64;

        if layer_idx == 0 && new_len == 1 && verbose() {
            eprintln!(
                "[CORRECTNESS-009] head_dim={}, using {} kernel, graph_mode={}, skip_debug={}",
                head_dim,
                if use_single_warp {
                    "single-warp"
                } else {
                    "multi-warp"
                },
                use_graph_mode,
                skip_debug
            );
        }

        let (kernel_type, module_key, config) = if use_single_warp {
            // Single-warp: 32 threads per head, no shared memory
            let ktype = KernelType::IncrementalAttention {
                max_seq_len: max_len as u32,
                head_dim: head_dim as u32,
                n_heads: num_heads as u32,
                n_kv_heads: num_kv_heads as u32,
                indirect: use_graph_mode,
            };
            let key = if use_graph_mode {
                format!(
                    "incremental_attention_indirect_{}_{}_{}_{}",
                    max_len, head_dim, num_heads, num_kv_heads
                )
            } else {
                format!(
                    "incremental_attention_{}_{}_{}_{}",
                    max_len, head_dim, num_heads, num_kv_heads
                )
            };
            // Grid: num_heads blocks, Block: 32 threads (1 warp)
            let cfg = LaunchConfig::grid_2d(num_heads as u32, 1, 32, 1);
            (ktype, key, cfg)
        } else {
            // Multi-warp: 128 threads per head (4 warps), uses shared memory
            // PAR-107-REVERTED: 8 warps SLOWER due to synchronization overhead
            // Five-Whys: More warps = more reduction barriers, hurts single-token decode
            let num_warps_per_head = 4;
            let ktype = KernelType::MultiWarpAttention {
                max_seq_len: max_len as u32,
                head_dim: head_dim as u32,
                n_heads: num_heads as u32,
                n_kv_heads: num_kv_heads as u32,
                num_warps_per_head,
                indirect: use_graph_mode,
            };
            let key = if use_graph_mode {
                format!(
                    "multi_warp_attention_indirect_{}_{}_{}_{}_{}",
                    max_len, head_dim, num_heads, num_kv_heads, num_warps_per_head
                )
            } else {
                format!(
                    "multi_warp_attention_{}_{}_{}_{}_{}",
                    max_len, head_dim, num_heads, num_kv_heads, num_warps_per_head
                )
            };
            // Grid: num_heads blocks, Block: 128 threads (4 warps)
            let cfg = LaunchConfig::grid_2d(num_heads as u32, 1, 32 * num_warps_per_head, 1);
            (ktype, key, cfg)
        };

        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let ptx = self.kernels.generate_ptx(&kernel_type);

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

        // PAR-074: Launch config already computed above in adaptive selection

        let mut ptr_q = q_gpu.as_ptr();
        let mut ptr_k = k_buf.as_ptr();
        let mut ptr_v = v_buf.as_ptr();
        let mut ptr_out = out_gpu.as_ptr();

        // PAR-069: Use graph mode (indirect kernel) ONLY when seq_len_buf is initialized
        let seq_len_arg: u64;
        if let Some(ref seq_len_buf) = self.seq_len_buf {
            // Graph capture mode - pass seq_len_buf pointer
            // CORRECTNESS-011: Use self.stream for graph capture (graph captures on stream, not compute_stream)
            let mut seq_len_ptr = seq_len_buf.as_ptr();
            seq_len_arg = seq_len_ptr;
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut ptr_q) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_k) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_v) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut seq_len_ptr) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        } else {
            // Normal mode - pass seq_len value directly
            // CORRECTNESS-012: Use self.stream (NOT compute_stream) to ensure synchronization
            // with subsequent GEMV operations which also use self.stream.
            // Five-Whys: GPU garbage output -> race condition -> attention on compute_stream,
            // output projection on stream -> no sync -> data corruption
            let mut seq_len_val = new_len as u32;
            seq_len_arg = seq_len_val as u64;
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
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
        }

        // trueno#243: Record kernel for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&module_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_q, ptr_k, ptr_v, ptr_out, seq_len_arg],
            });
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn incremental_attention_into_inner(
        &mut self,
        layer_idx: usize,
        q_gpu: &GpuBuffer<f32>,
        k_gpu: &GpuBuffer<f32>,
        v_gpu: &GpuBuffer<f32>,
        out_gpu: &GpuBuffer<f32>,
        skip_debug: bool,
    ) -> Result<usize, GpuError> {
        let num_heads = self.kv_num_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;

        // CORRECTNESS-013: Stateless GPU mode - disable KV cache to isolate cache bugs
        // When STATELESS_GPU=1, attention only sees the current token (no history)
        // If output becomes correct in stateless mode, the issue is in KV cache logic
        static STATELESS_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_stateless = *STATELESS_MODE.get_or_init(|| {
            let mode = std::env::var("STATELESS_GPU")
                .map(|v| v == "1")
                .unwrap_or(false);
            if mode {
                eprintln!("[CORRECTNESS-013] STATELESS_GPU mode ENABLED - attention only sees current token");
            }
            mode
        });

        // Get current cache length and check bounds
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        // CORRECTNESS-013: In stateless mode, always use seq_len=1 (only current token)
        let new_len = if use_stateless { 1 } else { cache_len + 1 };
        if !use_stateless && new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-051: KV cache overflow - max_len={}, trying to add position {}",
                max_len, new_len
            )));
        }

        // PAR-052: Use scatter kernel instead of per-head D2D copies
        // Replaces 2 * num_kv_heads D2D copies with 2 kernel launches
        self.scatter_kv_to_cache(layer_idx, k_gpu, v_gpu, cache_len, use_stateless)?;

        // Update cache length
        self.kv_cache_lengths.insert(layer_idx, new_len);

        // PAR-058-DEBUG: Trace attention parameters for layer 0 (only first 3 tokens)
        // PAR-054-FIX: Skip during graph capture to avoid sync breaking capture
        if !skip_debug && layer_idx == 0 && new_len <= 3 {
            let k_key = format!("kv_{}_k", layer_idx);
            let v_key = format!("kv_{}_v", layer_idx);
            let num_kv_heads = self.kv_num_kv_heads;
            self.debug_attention_trace(
                layer_idx, num_heads, num_kv_heads, head_dim, max_len, new_len,
                q_gpu, k_gpu, &k_key, &v_key,
            )?;
        }

        // PAR-118: Flash Decoding for split-K attention parallelism.
        // Five-Whys: Multi-warp uses only 28 blocks (one per head) = 22% SM occupancy on
        // RTX 4090. Flash Decoding splits KV scan across max_chunks blocks per head,
        // giving 28 * max_chunks = 224 blocks (8 chunks) = 175% more SM utilization.
        //
        // Set FLASH_DECODE=0 to disable (for debugging / A-B testing)
        let use_graph_mode = self.seq_len_buf.is_some();

        static FLASH_DECODE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let flash_enabled = *FLASH_DECODE_ENABLED.get_or_init(|| {
            !std::env::var("FLASH_DECODE")
                .map(|v| v == "0")
                .unwrap_or(false)
        });

        if flash_enabled
            && !self.is_prefilling
            && self.flash_decode_enabled
            && self.flash_decode_k_ptrs.contains_key(&layer_idx)
        {
            return self
                .flash_decoding_graphed(layer_idx, q_gpu, out_gpu, use_graph_mode, new_len as u32)
                .map(|()| new_len);
        }

        // Launch the attention kernel (single-warp or multi-warp, direct or indirect)
        self.launch_attention_kernel(layer_idx, q_gpu, out_gpu, new_len, use_graph_mode, skip_debug)?;

        // PAR-051: NO sync here - caller continues pipeline

        // CORRECTNESS-013: Debug attention output for layer 0 at seq_len=2
        if !skip_debug && layer_idx == 0 && new_len == 2 {
            self.stream.synchronize()?;
            let mut attn_out = vec![0.0f32; out_gpu.len()];
            out_gpu.copy_to_host(&mut attn_out)?;
            eprintln!(
                "[CORRECTNESS-013-ATTN] Layer 0 attention output at seq_len=2, first 10: {:?}",
                &attn_out[..10.min(attn_out.len())]
            );
            // Dump per-head output for first 3 heads
            for h in 0..3.min(num_heads) {
                let start = h * head_dim;
                eprintln!(
                    "[CORRECTNESS-013-ATTN] Head {} first 5: {:?}",
                    h,
                    &attn_out[start..start + 5]
                );
            }
        }

        Ok(new_len)
    }

    /// PAR-058-DEBUG: Trace attention inputs and KV cache for debugging.
    #[allow(clippy::too_many_arguments)]
    fn debug_attention_trace(
        &mut self,
        layer_idx: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_len: usize,
        new_len: usize,
        q_gpu: &GpuBuffer<f32>,
        k_gpu: &GpuBuffer<f32>,
        k_key: &str,
        v_key: &str,
    ) -> Result<(), GpuError> {
        self.compute_stream.synchronize()?;
        eprintln!(
            "[PAR-058-ATTN] Layer {}: num_heads={}, num_kv_heads={}, head_dim={}, max_len={}, seq_len={}",
            layer_idx, num_heads, num_kv_heads, head_dim, max_len, new_len
        );
        let mut k_input = vec![0.0f32; k_gpu.len()];
        k_gpu.copy_to_host(&mut k_input)?;
        let k_nan = k_input.iter().filter(|x| x.is_nan()).count();
        if k_nan > 0 {
            eprintln!("[PAR-058-ATTN] K input has {} NaN out of {}", k_nan, k_input.len());
        } else {
            eprintln!("[PAR-058-ATTN] K input OK, first 5: {:?}", &k_input[..5.min(k_input.len())]);
        }
        let mut q_input = vec![0.0f32; q_gpu.len()];
        q_gpu.copy_to_host(&mut q_input)?;
        let q_nan = q_input.iter().filter(|x| x.is_nan()).count();
        if q_nan > 0 {
            eprintln!("[PAR-058-ATTN] Q input has {} NaN out of {}", q_nan, q_input.len());
        } else {
            eprintln!("[PAR-058-ATTN] Q input OK, first 5: {:?}", &q_input[..5.min(q_input.len())]);
        }
        let k_cache = self.kv_cache_gpu.get(k_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("K cache not found for {k_key}"))
        })?;
        let cache_size = num_kv_heads * max_len * head_dim;
        let mut k_cache_vals = vec![0.0f32; cache_size];
        k_cache.copy_to_host(&mut k_cache_vals)?;
        let k_cache_nan = k_cache_vals.iter().filter(|x| x.is_nan()).count();
        if k_cache_nan > 0 {
            eprintln!("[PAR-058-ATTN] K cache has {} NaN", k_cache_nan);
        } else {
            eprintln!("[PAR-058-ATTN] K cache head0 pos0 first 5: {:?}", &k_cache_vals[..5.min(k_cache_vals.len())]);
            if new_len >= 2 && k_cache_vals.len() >= head_dim + 5 {
                eprintln!("[PAR-058-ATTN] K cache head0 pos1 first 5: {:?}", &k_cache_vals[head_dim..(head_dim + 5).min(k_cache_vals.len())]);
            }
        }
        let v_cache = self.kv_cache_gpu.get(v_key).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("V cache not found for {v_key}"))
        })?;
        let mut v_cache_vals = vec![0.0f32; cache_size];
        v_cache.copy_to_host(&mut v_cache_vals)?;
        let v_cache_nan = v_cache_vals.iter().filter(|x| x.is_nan()).count();
        if v_cache_nan > 0 {
            eprintln!("[PAR-058-ATTN] V cache has {} NaN", v_cache_nan);
        } else {
            eprintln!("[PAR-058-ATTN] V cache head0 pos0 first 5: {:?}", &v_cache_vals[..5.min(v_cache_vals.len())]);
        }
        Ok(())
    }
}
