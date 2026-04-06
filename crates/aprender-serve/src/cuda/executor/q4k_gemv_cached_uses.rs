impl CudaExecutor {
    // ========================================================================
    // PAR-005: Cached GEMV Methods (avoid per-call weight transfers)
    // ========================================================================

    /// Execute Q4_K GEMV using cached weights - PAR-005
    ///
    /// Uses pre-uploaded weights from `quantized_weight_cache` to avoid
    /// CPU→GPU transfer on every forward pass. Weights must be loaded
    /// beforehand via `load_quantized_weights()`.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight tensor
    /// * `input` - Input vector (f32, length k)
    /// * `output` - Output vector (f32, length n)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be divisible by 256)
    ///
    /// # Errors
    ///
    /// Returns error if weights not cached or kernel fails.
    pub fn q4k_gemv_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // PAR-057: Use TiledQ4KGemv for better performance (~4x fewer global reads)
        // Fall back to basic Q4KGemv if K not aligned to 256
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)
        let use_tiled = k.is_multiple_of(256) && k <= MAX_TILED_K;
        let use_chunked = k.is_multiple_of(256) && k > MAX_TILED_K;
        let outputs_per_block = 4u32;

        let (kernel_type, cache_key, config) = if use_chunked {
            // PAR-502: Use chunked kernel for large K dimensions (7B+ models)
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else if use_tiled {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            // NOTE: Shared memory is statically declared in PTX - do NOT pass dynamically
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else {
            let kt = KernelType::Q4KGemv { k, n };
            let ck = format!("q4k_gemv_{}_{}", k, n);
            let cfg = LaunchConfig::grid_2d(n, 1, 32, 1);
            (kt, ck, cfg)
        };

        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // GH-215 FIX: Pad activations to ceil(K/256)*256 when K not 256-aligned.
        // The Q4K kernel reads activations at sb_idx*256+val_idx, which reaches
        // up to (num_super_blocks-1)*256+255. Without padding, this is an OOB read.
        let padded_k = ((k as usize + 255) / 256) * 256;
        let padded_input: std::borrow::Cow<'_, [f32]> = if padded_k > input.len() {
            let mut padded = vec![0.0f32; padded_k];
            padded[..input.len()].copy_from_slice(input);
            std::borrow::Cow::Owned(padded)
        } else {
            std::borrow::Cow::Borrowed(input)
        };
        let buf_input = GpuBuffer::from_host(&self.context, &padded_input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = buf_input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-023: Execute Q4_K GEMV with GPU buffer input/output (async, no sync)
    ///
    /// This is the async variant that keeps data on GPU. Used for pipelining
    /// multiple operations without CPU round-trips.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight buffer
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    ///
    /// # Returns
    ///
    /// GPU buffer containing output vector (not synchronized)
    pub fn q4k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // CORRECTNESS-001: Use TiledQ4KGemv for aligned K (matches sync version)
        // The basic Q4KGemv kernel has the same scale extraction issue
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)
        let use_tiled = k.is_multiple_of(256) && k <= MAX_TILED_K;
        let use_chunked = k.is_multiple_of(256) && k > MAX_TILED_K;
        let outputs_per_block = 4u32;

        let (kernel_type, cache_key, config) = if use_chunked {
            // PAR-502: Use chunked kernel for large K dimensions (7B+ models)
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else if use_tiled {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else {
            let kt = KernelType::Q4KGemv { k, n };
            let ck = format!("q4k_gemv_{}_{}", k, n);
            let cfg = LaunchConfig::grid_2d(n, 1, 32, 1);
            (kt, ck, cfg)
        };

        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-023: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-058: Execute Q6_K GEMV using cached weight (async, no sync)
    ///
    /// Same as q4k_gemv_cached_async but for Q6_K quantized weights.
    /// Used for LM head when it's Q6K quantized.
    pub fn q6k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        let use_mwv = self.gpu_profile.q6k != crate::cuda::gpu_profile::Q6kVariant::Legacy && k.is_multiple_of(256);
        let num_warps = self.gpu_profile.mwv_warps;

        let (kernel_type, cache_key, config) = if use_mwv {
            let kt = KernelType::MwvQ6KGemv { k, n, num_warps };
            let ck = format!("mwv_q6k_gemv_{}_{}_{}", k, n, num_warps);
            let cfg = LaunchConfig::grid_2d(n, 1, num_warps * 32, 1);
            (kt, ck, cfg)
        } else {
            let kt = KernelType::Q6KGemv { k, n };
            let ck = format!("q6k_gemv_{}_{}", k, n);
            let cfg = LaunchConfig::grid_2d(n, 1, 32, 1);
            (kt, ck, cfg)
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-058: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-043: Execute Q4_K GEMV using pre-indexed device pointer (async, no sync)
    ///
    /// This eliminates HashMap lookup + string formatting overhead (~10ms per token).
    /// Weight pointer must be from `indexed_layer_weights` populated by `build_indexed_weights()`.
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q4k_gemv_indexed_async(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Validate pointer before kernel launch — launching with ptr=0
        // crashes the kernel and permanently poisons the CUDA context.
        if weight_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "null weight pointer in q4k_gemv_indexed_async".to_string(),
            ));
        }

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // PAR-082-V2: Use MwvQ4KGemv with configurable warp count
        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::MwvQ4KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("mwv_q4k_gemv_{}_{}_{}", k, n, num_warps);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // num_warps * 32 threads per output element
        let threads = num_warps * 32;
        let config = LaunchConfig::grid_2d(n, 1, threads, 1);
        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(buf_output)
    }
}
