impl CudaExecutor {

    /// PAR-058: Execute Q6_K GEMV using pre-indexed device pointer (async, no sync)
    ///
    /// Like `q4k_gemv_indexed_async` but for Q6_K quantized weights.
    /// Used when V projection weights are Q6_K quantized (some GGUF models).
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q6K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q6k_gemv_indexed_async(
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
                "null weight pointer in q6k_gemv_indexed_async".to_string(),
            ));
        }
        use crate::cuda::gpu_profile::Q6kVariant;
        let num_warps = self.gpu_profile.mwv_warps;

        // Dispatch Q6K variant from GpuProfile (auto-detected or env var override)
        let can_use_advanced = k.is_multiple_of(256);

        if can_use_advanced && self.gpu_profile.q6k == Q6kVariant::HwDp4a {
            let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;
            self.hw_dp4a_q6k_gemv_into(weight_ptr, input, &buf_output, n, k)?;
            return Ok(buf_output);
        }

        if can_use_advanced && self.gpu_profile.q6k == Q6kVariant::Dp4a {
            let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;
            self.dp4a_q6k_gemv_into(weight_ptr, input, &buf_output, n, k)?;
            return Ok(buf_output);
        }

        let (kernel_type, cache_key, config) = if can_use_advanced && self.gpu_profile.q6k == Q6kVariant::Mwv {
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

        Ok(buf_output)
    }

    /// PAR-044: Execute Q4_K GEMV into existing buffer (zero-allocation, async)
    ///
    /// Dispatches to optimal kernel variant from auto-detected GpuProfile.
    /// Default: HwDp4a on sm_75+ (Turing/Ampere/Ada/Hopper), Mwv otherwise.
    /// Override with env vars (HW_DP4A_Q4K, DP4A_Q4K, etc.) for experimentation.
    #[inline]
    pub fn q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        use crate::cuda::gpu_profile::Q4kVariant;
        match self.gpu_profile.q4k {
            Q4kVariant::Legacy => self.q4k_gemv_into_legacy(weight_ptr, input, output, n, k),
            Q4kVariant::Wide => self.wide_q4k_gemv_into(weight_ptr, input, output, n, k),
            Q4kVariant::Vectorized => self.vectorized_q4k_gemv_into(weight_ptr, input, output, n, k),
            Q4kVariant::MwvDp4a => self.mwv_dp4a_q4k_gemv_into(weight_ptr, input, output, n, k),
            Q4kVariant::HwDp4a => self.hw_dp4a_q4k_gemv_into(weight_ptr, input, output, n, k),
            Q4kVariant::Mwv => self.mwv_q4k_gemv_into(weight_ptr, input, output, n, k),
        }
    }

    /// Legacy Q4K GEMV dispatch (pre-PAR-132)
    /// Uses tiled/chunked kernels with 128 threads or basic with 32 threads
    fn q4k_gemv_into_legacy(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q4k_gemv_into_legacy")?;
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288;
        let use_tiled = k.is_multiple_of(256) && k <= MAX_TILED_K;
        let use_chunked = k.is_multiple_of(256) && k > MAX_TILED_K;
        let outputs_per_block = 4u32;

        let (kernel_type, cache_key, config) = if use_chunked {
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

        let mut ptr_output = output.as_ptr();
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

        Ok(())
    }

    /// PAR-062: Execute Coalesced Q4_K GEMV with bandwidth-optimized memory access
    ///
    /// Key optimizations over basic Q4KGemvKernel:
    /// 1. **Scale loading**: Lane 0 loads 12 scale bytes as 3 x u32, broadcasts via shuffle
    ///    - Reduces 384 redundant byte loads to 3 loads + 3 broadcasts per super-block
    /// 2. **Reduced memory transactions**: Better cache utilization
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    #[inline]
    pub fn coalesced_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "coalesced_q4k_gemv_into")?;
        let kernel_type = KernelType::CoalescedQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("coalesced_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // One warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = output.as_ptr();
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

        Ok(())
    }

    /// PAR-132: Wide Q4_K GEMV with 256 threads (8 warps) per output
    ///
    /// Root cause fix for 3x Ollama performance gap:
    /// - Previous: 32 threads/block = 33% SM occupancy, can't hide memory latency
    /// - New: 256 threads/block = 67-100% occupancy, 8 warps hide latency
    ///
    /// Cross-warp reduction via 32 bytes shared memory per block.
    /// Target: 100+ tok/s decode (from 39 tok/s), reaching Ollama parity.
    #[inline]
    pub fn wide_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "wide_q4k_gemv_into")?;
        let kernel_type = KernelType::WideQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("wide_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-132: 8 warps (256 threads) per output element
        // Empirically: 8 warps (67 tok/s) > 4 warps (61 tok/s) on RTX 4090
        let config = LaunchConfig::grid_2d(n, 1, 256, 1);

        let mut ptr_output = output.as_ptr();
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

        Ok(())
    }

    /// PAR-069: Execute Vectorized Q4_K GEMV with coalesced u32 weight loads
    ///
    /// Key optimization over CoalescedQ4KGemv:
    /// 1. **Weight loading**: Uses ld_global_u32 for coalesced 4-byte loads
    ///    - 32 threads × 4 bytes = 128 bytes per transaction (vs 32 × 1 byte scattered)
    /// 2. **Memory bandwidth**: Target 80%+ of peak (vs 6% with byte loads)
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    #[inline]
    pub fn vectorized_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "vectorized_q4k_gemv_into")?;
        let kernel_type = KernelType::VectorizedQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("vectorized_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // One warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = output.as_ptr();
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

        Ok(())
    }

}
