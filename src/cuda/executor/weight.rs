impl CudaExecutor {
    /// Batched Q6K GEMV (matrix-vector multiply) with quantized weights.
    ///
    /// Performs `output = weight * input` where weight is Q6K quantized.
    pub fn batched_q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "batched_q6k_gemv_into")?;
        debug_assert!(
            k.is_multiple_of(256),
            "K must be multiple of 256 for Q6K super-blocks"
        );

        let kernel_type = KernelType::BatchedQ6KGemv { k, n, m };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("batched_q6k_gemv_{}_{}_{}", m, k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: N blocks (one per output row), 32 threads per block
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;
        let mut m_val = m;

        // Kernel signature: batched_q6k_gemv_warp_reduce(y_ptr, w_ptr, x_ptr, k_dim, n_dim, m_dim)
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
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-058: Execute Q6_K GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q4k_gemv_into` but for Q6_K quantized weights.
    /// Used when V projection weights are Q6_K quantized (some GGUF models).
    ///
    /// Q6_K format: 210 bytes per 256 elements (vs Q4_K's 144 bytes)
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q6K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q6k_gemv_into")?;
        use crate::cuda::gpu_profile::Q6kVariant;
        let can_use_advanced = k.is_multiple_of(256);
        if can_use_advanced && self.gpu_profile.q6k == Q6kVariant::HwDp4a {
            return self.hw_dp4a_q6k_gemv_into(weight_ptr, input, output, n, k);
        }
        if can_use_advanced && self.gpu_profile.q6k == Q6kVariant::Dp4a {
            return self.dp4a_q6k_gemv_into(weight_ptr, input, output, n, k);
        }
        if can_use_advanced && self.gpu_profile.q6k == Q6kVariant::Mwv {
            return self.mwv_q6k_gemv_into(weight_ptr, input, output, n, k);
        }
        // Original Q6K kernel (CoalescedQ6K disabled due to CORRECTNESS-006)
        let kernel_type = KernelType::Q6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q6k_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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

        // trueno#243: Record kernel for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_output, ptr_weights, ptr_input, k_val as u64, n_val as u64],
            });
        }

        Ok(())
    }

    /// GH-118: Multi-warp Q6K GEMV for Orin decode throughput (Design by Contract)
    ///
    /// Contracts:
    /// - Precondition: k % 256 == 0 (Q6K super-block alignment)
    /// - Precondition: weight_ptr != 0 (valid device pointer)
    /// - Postcondition: output[i] == Q6KGemvKernel output[i] for all i (parity)
    /// - Invariant: PARITY-114 barrier safety verified at PTX generation time
    ///
    /// Auto-selected by GpuProfile on sm < 7.5 (no DP4A). Uses gpu_profile.mwv_warps.
    #[inline]
    pub fn mwv_q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "mwv_q6k_gemv_into")?;
        debug_assert!(
            k.is_multiple_of(256),
            "K must be multiple of 256 for Q6K super-blocks"
        );
        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::MwvQ6KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("mwv_q6k_gemv_{}_{}_{}", k, n, num_warps);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let threads = num_warps * 32;
        let config = LaunchConfig::grid_2d(n, 1, threads, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All device pointers are allocated by CudaExecutor and valid for
        // the kernel's grid dimensions. k_val and n_val are stack-local scalars.
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

        // trueno#243: Record kernel for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_output, ptr_weights, ptr_input, k_val as u64, n_val as u64],
            });
        }

        Ok(())
    }

    /// DP4A Q6_K GEMV with vectorized int32 loads and dp4a.u32.s32
    ///
    /// Two-step pipeline (same pattern as DP4A Q4K):
    /// 1. Quantize f32 activations → Q8_1 format
    /// 2. DP4A dot product: Q6K weights × Q8_1 activations → f32 output
    ///
    /// Enable with `DP4A_Q6K=1` env var.
    pub fn dp4a_q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "dp4a_q6k_gemv_into")?;

        // Borrow pre-allocated Q8 buffer from workspace
        let q8_ptr = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("dp4a_q6k: workspace.q8_activation_buf not initialized")
            .as_ptr();
        let q8_len = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("q8_activation_buf must be initialized")
            .len();

        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };

        // Step 1: Quantize activations to Q8_1 (skip if already valid — PMAT-027)
        if !self.q8_activation_valid {
            self.q8_quantize_into(input, &q8_buf, k)?;
            self.q8_activation_valid = true;
        }

        // Step 2: Launch DP4A Q6K GEMV kernel
        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::Dp4aQ6KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("dp4a_q6k_gemv_{}_{}_{}", k, n, num_warps);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let threads = num_warps * 32;
        // Scale grid to GPU size for memory latency hiding.
        // Grid-stride loop in kernel handles the remainder.
        let grid_x = n.min(self.num_sms * 16);
        let config = LaunchConfig::grid_2d(grid_x, 1, threads, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_q8 = q8_buf.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_q8) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
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
                arg_data: vec![ptr_output, ptr_weights, ptr_q8, k_val as u64, n_val as u64],
            });
        }

        std::mem::forget(q8_buf);

        Ok(())
    }

    /// PMAT-030: Half-warp DP4A Q6K GEMV — 16 threads/SB, direct scale loads.
    #[inline]
    pub fn hw_dp4a_q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "hw_dp4a_q6k_gemv_into")?;

        let q8_ptr = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("hw_dp4a_q6k: workspace.q8_activation_buf not initialized")
            .as_ptr();
        let q8_len = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("q8_activation_buf must be initialized")
            .len();

        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };

        if !self.q8_activation_valid {
            self.q8_quantize_into(input, &q8_buf, k)?;
            self.q8_activation_valid = true;
        }

        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::HwDp4aQ6KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("hw_dp4a_q6k_gemv_{}_{}_{}", k, n, num_warps);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let threads = num_warps * 32;
        let grid_x = n.min(self.num_sms * 16);
        let config = LaunchConfig::grid_2d(grid_x, 1, threads, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_q8 = q8_buf.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_q8) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
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
                arg_data: vec![ptr_output, ptr_weights, ptr_q8, k_val as u64, n_val as u64],
            });
        }

        std::mem::forget(q8_buf);

        Ok(())
    }

    /// PAR-066: Execute coalesced Q6K GEMV into existing buffer
    ///
    /// Uses vectorized scale loading (4 x u32) instead of 16 single-byte loads.
    /// Five-Whys root cause: Original Q6KGemvKernel caused 16 memory transactions
    /// per super-block for scale loading. This kernel reduces to 4 transactions.
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q6K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    #[inline]
    pub fn coalesced_q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "coalesced_q6k_gemv_into")?;
        let kernel_type = KernelType::CoalescedQ6KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("coalesced_q6k_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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

    /// PAR-058: Execute Q8_0 GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q4k_gemv_into` but for Q8_0 quantized weights.
    /// Used when FFN down weights are Q8_0 quantized (some GGUF models like Qwen2.5-0.5B).
    ///
    /// Q8_0 format: 34 bytes per 32 elements (2-byte fp16 scale + 32 int8 values)
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q8_0 weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q8_0_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q8_0_gemv_into")?;
        // PAR-058: Zero allocation Q8_0 GEMV for mixed-quantization models
        let kernel_type = KernelType::Q8_0Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q8_0_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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

        // trueno#243: Record kernel for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_output, ptr_weights, ptr_input, k_val as u64, n_val as u64],
            });
        }

        Ok(())
    }

    /// PAR-058: Execute Q5_0 GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q8_0_gemv_into` but for Q5_0 quantized weights.
    /// Used when Q/K weights are Q5_0 quantized (Qwen 0.5B).
    ///
    /// Q5_0 format: 22 bytes per 32 elements (2-byte fp16 scale + 4-byte high bits + 16 bytes packed nibbles)
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q5_0 weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q5_0_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q5_0_gemv_into")?;
        // PAR-058: Zero allocation Q5_0 GEMV for Qwen 0.5B Q/K weights
        let kernel_type = KernelType::Q5_0Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5_0_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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

        // trueno#243: Record kernel for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_output, ptr_weights, ptr_input, k_val as u64, n_val as u64],
            });
        }

        Ok(())
    }

    /// PAR-058: Execute Q4_0 GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q5_0_gemv_into` but for Q4_0 quantized weights.
    /// Used when GGUF header claims Q5_0 but data is actually Q4_0 format (qtype mismatch).
    ///
    /// Q4_0 format: 18 bytes per 32 elements (2-byte fp16 scale + 16 bytes packed nibbles)
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4_0 weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q4_0_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q4_0_gemv_into")?;
        // PAR-058: Zero allocation Q4_0 GEMV for GGUF qtype mismatch
        let kernel_type = KernelType::Q4_0Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4_0_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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

        // trueno#243: Record kernel for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_output, ptr_weights, ptr_input, k_val as u64, n_val as u64],
            });
        }

        Ok(())
    }

    /// GH-374: Execute F32 GEMV into existing buffer (zero-allocation, async)
    ///
    /// CONTRACT: weight_ptr points to N*K f32 values (4 bytes each).
    /// Uses KernelType::Gemv which maps to trueno-gpu's GemvKernel (gemv_warp_reduce).
    ///
    /// This is needed for APR checkpoints where the LM head is stored as F32
    /// (GGML type 0) rather than a quantized format. Without this, F32 weights
    /// were silently misclassified as Q4K, producing garbage logits.
    pub fn f32_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "f32_gemv_into")?;
        let kernel_type = KernelType::Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("f32_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

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

        // SAFETY: All pointers validated above, kernel params match signature
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

        // trueno#243: Record kernel for manual graph construction
        if self.graph_recording {
            let module = self.modules.get_mut(&cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_output, ptr_weights, ptr_input, k_val as u64, n_val as u64],
            });
        }

        Ok(())
    }
}
