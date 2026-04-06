impl CudaExecutor {
    /// PAR-041: Execute Tiled Q4_K GEMV with shared memory caching (async, no sync)
    ///
    /// This variant uses 256 threads per block (vs 32 in q4k_gemv_cached_async) for
    /// better GPU occupancy. Input vector is cached in shared memory and shared by
    /// multiple output computations.
    ///
    /// Performance improvement: ~8x fewer global memory reads for input vector.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight buffer
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    /// * `outputs_per_block` - Number of outputs computed per block (default: 4)
    ///
    /// # Returns
    ///
    /// GPU buffer containing output vector (not synchronized)
    pub fn tiled_q4k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
        outputs_per_block: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)

        // Load kernel module - select based on K size to avoid shared memory overflow
        let (kernel_type, cache_key) = if k > MAX_TILED_K && k.is_multiple_of(256) {
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            (kt, ck)
        } else {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            (kt, ck)
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

        // PAR-041: Grid configuration for tiled kernel
        // ceil(N / outputs_per_block) blocks, (32 * outputs_per_block) threads per block
        // CRITICAL: Thread count must match kernel's load stride of 32 * outputs_per_block
        // NOTE: Shared memory is statically declared in PTX - do NOT pass dynamically
        let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
        let threads_per_block = 32 * outputs_per_block; // 4 outputs = 128 threads
        let config = LaunchConfig::grid_2d(num_blocks, 1, threads_per_block, 1);

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

        // PAR-041: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-056: Execute Chunked Tiled Q4_K GEMV for large K dimensions (async, no sync)
    ///
    /// This kernel handles K > 8192 where TiledQ4KGemvKernel's shared memory
    /// would exceed CUDA limits (48KB default, 96KB max). It processes the
    /// input vector in 32KB (8K float) chunks.
    ///
    /// Used for 7B+ model FFN down projection where K = intermediate_dim > 8K.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight buffer
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    /// * `outputs_per_block` - Number of outputs computed per block (default: 4)
    ///
    /// # Returns
    ///
    /// GPU buffer containing output vector (not synchronized)
    pub fn chunked_tiled_q4k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
        outputs_per_block: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // Load kernel module
        let kernel_type = KernelType::ChunkedTiledQ4KGemv {
            k,
            n,
            outputs_per_block,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);

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

        // PAR-056: Grid configuration for chunked tiled kernel
        // ceil(N / outputs_per_block) blocks, (32 * outputs_per_block) threads per block
        // CRITICAL: Thread count must match kernel's load stride of 32 * outputs_per_block
        // NOTE: Shared memory (32KB fixed) is statically declared in PTX - do NOT pass dynamically
        let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
        let threads_per_block = 32 * outputs_per_block; // 4 outputs = 128 threads
        let config = LaunchConfig::grid_2d(num_blocks, 1, threads_per_block, 1);

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

        // PAR-056: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-063: Execute DP4A Q4_K GEMV for optimized instruction throughput (async, no sync)
    ///
    /// Uses DP4A SIMD instruction to compute 4 multiply-adds per instruction,
    /// achieving up to 4x instruction reduction over scalar FMA operations.
    ///
    /// Key optimizations:
    /// - Vectorized scale loading (3 x u32 + warp shuffle broadcast)
    /// - DP4A instruction for SIMD dot products
    /// - Expected 2.5-3x throughput improvement over TiledQ4KGemv
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight buffer
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    ///
    /// # Returns
    ///
    /// GPU buffer containing output vector (not synchronized)
    pub fn dp4a_q4k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // Load kernel module
        let kernel_type = KernelType::Dp4aQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("dp4a_q4k_gemv_{}_{}", k, n);

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

        // PAR-063: Grid configuration - one warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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

        // PAR-063: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-063-V4: Quantize f32 activations to Q8_1 format (async, no sync)
    ///
    /// This is the first step in the true DP4A GEMV pipeline:
    /// 1. Q8 quantize: f32 → Q8_1 (this function)
    /// 2. Q4K×Q8 dot: Q4K weights × Q8_1 activations → f32 output
    ///
    /// Q8_1 format: 36 bytes per 32 values
    /// - 32 bytes: 32 × int8 quantized values (qs)
    /// - 2 bytes: fp16 scale
    /// - 2 bytes: fp16 sum (for bias correction)
    ///
    /// # Arguments
    /// * `input` - GPU buffer containing f32 activations
    /// * `n` - Number of elements to quantize
    ///
    /// # Returns
    /// GPU buffer containing Q8_1 quantized data (ceil(n/32) * 36 bytes)
    pub fn q8_quantize_async(
        &mut self,
        input: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<GpuBuffer<u8>, GpuError> {
        // Load kernel module
        let kernel_type = KernelType::Q8Quantize { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q8_quantize_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Q8_1 format: 36 bytes per 32 values
        // Layout: [qs: 32 × u8][scale: f16][sum: f16]
        let num_blocks = (n + 31) / 32;
        let output_bytes = (num_blocks * 36) as usize;
        let buf_output = GpuBuffer::<u8>::new(&self.context, output_bytes)?;

        // One warp (32 threads) processes 32 f32 values into one Q8_1 block
        let config = LaunchConfig::grid_2d(num_blocks, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_input = input.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(buf_output)
    }

    /// PAR-PERF-DP4A: Q8 quantize into PRE-ALLOCATED buffer (zero allocation)
    ///
    /// Five-Whys root cause (2026-02-09):
    /// 1. Why is DP4A path 10.7 tok/s (8x slower than MWV)?
    /// 2. Why so slow? → q8_quantize_async allocates a new GpuBuffer per call
    /// 3. Why per call? → GpuBuffer::new calls cudaMalloc (10-50us each)
    /// 4. Why 280x per token? → Called for every GEMV (10/layer × 28 layers)
    /// 5. ROOT CAUSE: No pre-allocated Q8 buffer in workspace
    /// FIX: Use workspace.q8_activation_buf (allocated once at init)
    pub fn q8_quantize_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<u8>,
        n: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::Q8Quantize { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q8_quantize_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let num_blocks = (n + 31) / 32;
        let config = LaunchConfig::grid_2d(num_blocks, 1, 32, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_input = input.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // trueno#243: Record kernel for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_output, ptr_input, n_val as u64],
            });
        }

        Ok(())
    }
}
