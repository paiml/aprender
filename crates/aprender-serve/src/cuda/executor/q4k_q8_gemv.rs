impl CudaExecutor {

    /// PAR-063-V5: Q4K × Q8 GEMV using true integer DP4A (async, no sync)
    ///
    /// This is the second step in the true DP4A GEMV pipeline:
    /// 1. Q8 quantize: f32 → Q8_1 (use q8_quantize_async)
    /// 2. Q4K×Q8 dot: Q4K weights × Q8_1 activations → f32 output (this function)
    ///
    /// Uses dp4a.u32.s32 instruction: d = dot4(weights_u8, activations_s8) + acc
    /// This achieves 4 multiply-adds per instruction vs 1 for scalar FMA.
    ///
    /// # Arguments
    /// * `weight_name` - Name of cached Q4K weight
    /// * `q8_input` - Q8_1 quantized activations from q8_quantize_async
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    pub fn q4k_q8_gemv_async(
        &mut self,
        weight_name: &str,
        q8_input: &GpuBuffer<u8>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // Load kernel module
        let kernel_type = KernelType::Q4KQ8Dot { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_q8_dot_{}_{}", k, n);

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

        // One warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_q8_input = q8_input.as_ptr();
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
                    std::ptr::from_mut(&mut ptr_q8_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(buf_output)
    }

    /// PAR-063-V5: Fused Q8 quantize + Q4K×Q8 GEMV (async, no sync)
    ///
    /// Combines both steps of the true DP4A pipeline into a single call:
    /// 1. Quantizes f32 activations to Q8_1
    /// 2. Computes Q4K × Q8_1 dot product using integer DP4A
    ///
    /// This is the drop-in replacement for dp4a_q4k_gemv_cached_async that
    /// achieves true 4x instruction reduction via integer arithmetic.
    ///
    /// # Arguments
    /// * `weight_name` - Name of cached Q4K weight
    /// * `input` - GPU buffer containing f32 activations
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    pub fn true_dp4a_q4k_gemv_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Step 1: Quantize activations to Q8_1
        let q8_activations = self.q8_quantize_async(input, k)?;

        // Step 2: Q4K × Q8 dot product
        self.q4k_q8_gemv_async(weight_name, &q8_activations, n, k)
    }

    /// PAR-063-V6: Packed DP4A Q4K×Q8 GEMV using true dp4a.u32.s32 instruction
    ///
    /// Key optimizations over Q4KQ8DotKernel:
    /// - Uses dp4a.u32.s32 to process 4 values per instruction (4x IPC)
    /// - Packs 4 Q4K nibbles into u32 for DP4A weight operand
    /// - Packs 4 Q8 values into u32 for DP4A activation operand
    /// - 2 DP4A calls per thread per super-block (8 values total)
    ///
    /// Expected speedup: 4x vs scalar Q4KQ8DotKernel
    pub fn packed_dp4a_q4k_q8_gemv_async(
        &mut self,
        weight_name: &str,
        q8_input: &GpuBuffer<u8>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // Load kernel module
        let kernel_type = KernelType::PackedDp4aQ4KQ8 { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("packed_dp4a_q4k_q8_{}_{}", k, n);

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

        // One warp (32 threads) per output element
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = buf_output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_q8_input = q8_input.as_ptr();
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
                    std::ptr::from_mut(&mut ptr_q8_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(buf_output)
    }

    /// PAR-063-V6: Fused packed DP4A Q4K×Q8 GEMV (quantize + compute)
    ///
    /// Combines:
    /// 1. f32 → Q8_1 quantization
    /// 2. Packed DP4A Q4K×Q8 dot product
    ///
    /// This is the highest-performance path for Q4_K inference.
    pub fn packed_dp4a_full_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Step 1: Quantize activations to Q8_1
        let q8_activations = self.q8_quantize_async(input, k)?;

        // Step 2: Packed DP4A Q4K × Q8 dot product
        self.packed_dp4a_q4k_q8_gemv_async(weight_name, &q8_activations, n, k)
    }

    /// Execute Q5_K GEMV using cached weights - PAR-005
    pub fn q5k_gemv_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        let kernel_type = KernelType::Q5KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5k_gemv_{}_{}", k, n);

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
        // The Q5K kernel reads activations at sb_idx*256+val_idx, which reaches
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

        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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

    /// Execute Q6_K GEMV using cached weights - PAR-005
    pub fn q6k_gemv_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        let kernel_type = KernelType::Q6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q6k_gemv_{}_{}", k, n);

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
        // The Q6K kernel reads activations at sb_idx*256+val_idx, which reaches
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

        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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

    /// PAR-014: Apply GELU activation in-place on a GPU buffer
    ///
    /// Uses BiasActivation kernel with zero bias for pure GELU.
    /// Part of persistent GPU tensor optimization for M4 milestone.
    pub fn gelu_gpu(&mut self, buffer: &GpuBuffer<f32>, n: u32) -> Result<(), GpuError> {
        // Use BiasActivation kernel with GELU activation (type 2) and zero bias
        let kernel_type = KernelType::BiasActivation {
            n,
            bias_size: 1,  // Single zero element
            activation: 2, // GELU
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gelu_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Zero bias buffer (single element)
        let zero_bias = GpuBuffer::from_host(&self.context, &[0.0f32])?;

        // Launch config: 256 threads per block, enough blocks to cover n elements
        let threads_per_block = 256u32;
        let blocks = (n + threads_per_block - 1) / threads_per_block;
        let config = LaunchConfig::grid_2d(blocks, 1, threads_per_block, 1);

        let mut ptr_output = buffer.as_ptr();
        let mut ptr_bias = zero_bias.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_bias) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // No sync - caller can batch operations
        Ok(())
    }
}
