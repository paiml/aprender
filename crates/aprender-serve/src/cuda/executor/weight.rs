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

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

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
        // PMAT-806: Q6K is deliberately NOT outlier-rerouted. On-device sweep
        // (gx10 GB10) showed the fp32 MWV-Q6K path diverges on this model's
        // lm_head (cosine gate FAILs → wgpu fallback), while routing only the
        // Q4K activation-quant GEMVs to fp32 MWV already lifts the gate to
        // 0.9939 (Q6K's HwDp4a error on the residual is tolerable). Keeping Q6K
        // on HwDp4a avoids the MWV-Q6K defect (tracked separately).
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

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

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

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

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

        // SAFETY: constructs a non-owning `GpuBuffer` view over an already-allocated device region (`ptr`, element count `len`) that stays live for the kernel call; the view is `leak()`ed afterwards so its Drop never frees the borrowed device allocation (no double-free).
        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };

        // Step 1: Quantize activations to Q8_1 (skip if already valid — PMAT-027)
        self.q8_quantize_cached(input, &q8_buf, k)?;

        // Step 2: Launch DP4A Q6K GEMV kernel
        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::Dp4aQ6KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("dp4a_q6k_gemv_{}_{}_{}", k, n, num_warps);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

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

        // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature, and every referenced device buffer is allocated, correctly sized, and lives until the stream-ordered launch completes.
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

        // SAFETY: constructs a non-owning `GpuBuffer` view over an already-allocated device region (`ptr`, element count `len`) that stays live for the kernel call; the view is `leak()`ed afterwards so its Drop never frees the borrowed device allocation (no double-free).
        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };

        self.q8_quantize_cached(input, &q8_buf, k)?;

        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::HwDp4aQ6KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("hw_dp4a_q6k_gemv_{}_{}_{}", k, n, num_warps);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

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

        // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature, and every referenced device buffer is allocated, correctly sized, and lives until the stream-ordered launch completes.
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

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

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

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

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

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

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

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

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

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

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

    /// #3477: Execute IQ4_XS GEMV into an existing buffer (zero-allocation, async).
    ///
    /// CONTRACT: `weight_ptr` points at `N * ceil(K/256)` super-blocks of 136
    /// bytes, ROW-MAJOR — row `i` starts at `weight_ptr + i * nb * 136`.
    /// LAYOUT-001 applies exactly as for every other GEMV here.
    ///
    /// The block layout is a port of `quantize::iq4_xs::dequantize_iq4_xs_block`,
    /// which is itself checked against llama.cpp and gguf-py. That function is
    /// the oracle: a disagreement is this kernel's bug.
    pub fn iq4_xs_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "iq4_xs_gemv_into")?;
        let kernel_type = KernelType::Iq4XsGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("iq4_xs_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: pointers validated above; params match the PTX entry signature.
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

    /// #3869: Execute IQ4_NL GEMV into an existing buffer (zero-allocation, async).
    ///
    /// CONTRACT: `weight_ptr` points at `N * ceil(K/32)` blocks of **18** bytes,
    /// ROW-MAJOR - row `i` starts at `weight_ptr + i * nb * 18`. Note the 32,
    /// not 256: IQ4_NL is the one type here whose block is not a super-block.
    /// LAYOUT-001 applies exactly as for every other GEMV in this file.
    ///
    /// The block layout is a port of `quantize::iq4_nl::dequantize_iq4_nl_block`,
    /// which is checked against llama.cpp's `dequantize_row_iq4_nl` and, more
    /// usefully, against the independently transcribed IQ4_XS decoder. That
    /// function is the oracle: a disagreement is this kernel's bug.
    pub fn iq4_nl_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "iq4_nl_gemv_into")?;
        let kernel_type = KernelType::Iq4NlGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("iq4_nl_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: pointers validated above; params match the PTX entry signature.
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

    /// #3884: Execute IQ3_S GEMV into an existing buffer (zero-allocation, async).
    ///
    /// CONTRACT: `weight_ptr` points at `N * ceil(K/256)` super-blocks of 110
    /// bytes, ROW-MAJOR - row `i` starts at `weight_ptr + i * nb * 110`.
    /// LAYOUT-001 applies as for every other GEMV in this file.
    ///
    /// The block layout is a port of `quantize::iq3_s::dequantize_iq3_s_block`,
    /// itself checked against llama.cpp and gguf-py. That function is the
    /// oracle: a disagreement is this kernel's bug.
    pub fn iq3_s_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "iq3_s_gemv_into")?;
        let kernel_type = KernelType::Iq3SGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("iq3_s_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: pointers validated above; params match the PTX entry signature.
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

    /// #3960: Execute Q2_K GEMV into an existing buffer (zero-allocation, async).
    ///
    /// CONTRACT: `weight_ptr` points at `N * ceil(K/256)` super-blocks of 84
    /// bytes, ROW-MAJOR - row `i` starts at `weight_ptr + i * nb * 84`.
    /// LAYOUT-001 applies as for every other GEMV in this file.
    ///
    /// The block layout is a port of `quantize::dequant::dequantize_q2_k` (proven bitwise against gguf-py, #3960),
    /// itself checked against llama.cpp and gguf-py. That function is the
    /// oracle: a disagreement is this kernel's bug.
    pub fn q2_k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q2_k_gemv_into")?;
        let kernel_type = KernelType::Q2KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q2_k_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: pointers validated above; params match the PTX entry signature.
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

    /// #3931: Execute IQ2_XXS GEMV into an existing buffer (zero-allocation, async).
    ///
    /// CONTRACT: `weight_ptr` points at `N * ceil(K/256)` super-blocks of 66
    /// bytes, ROW-MAJOR - row `i` starts at `weight_ptr + i * nb * 66`.
    /// LAYOUT-001 applies as for every other GEMV in this file.
    ///
    /// The block layout is a port of `quantize::iq2_xxs::dequantize_iq2_xxs_block`,
    /// itself checked against llama.cpp and gguf-py. That function is the
    /// oracle: a disagreement is this kernel's bug.
    pub fn iq2_xxs_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "iq2_xxs_gemv_into")?;
        let kernel_type = KernelType::Iq2XxsGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("iq2_xxs_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: pointers validated above; params match the PTX entry signature.
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

    /// #3953: Execute IQ2_S GEMV into an existing buffer (zero-allocation, async).
    ///
    /// CONTRACT: `weight_ptr` points at `N * ceil(K/256)` super-blocks of 82
    /// bytes, ROW-MAJOR - row `i` starts at `weight_ptr + i * nb * 82`.
    /// LAYOUT-001 applies as for every other GEMV in this file.
    ///
    /// The block layout is a port of `quantize::iq2_s::dequantize_iq2_s_block`,
    /// itself checked against llama.cpp and gguf-py. That function is the
    /// oracle: a disagreement is this kernel's bug.
    pub fn iq2_s_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "iq2_s_gemv_into")?;
        let kernel_type = KernelType::Iq2SGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("iq2_s_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: pointers validated above; params match the PTX entry signature.
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

    /// #3963: Execute IQ3_XXS GEMV into an existing buffer (zero-allocation, async).
    ///
    /// CONTRACT: `weight_ptr` points at `N * ceil(K/256)` super-blocks of 98
    /// bytes, ROW-MAJOR - row `i` starts at `weight_ptr + i * nb * 98`.
    /// LAYOUT-001 applies as for every other GEMV in this file.
    ///
    /// The block layout is a port of `quantize::iq3_xxs::dequantize_iq3_xxs_block`,
    /// itself checked against llama.cpp and gguf-py. That function is the
    /// oracle: a disagreement is this kernel's bug.
    pub fn iq3_xxs_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "iq3_xxs_gemv_into")?;
        let kernel_type = KernelType::Iq3XxsGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("iq3_xxs_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: pointers validated above; params match the PTX entry signature.
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

    /// #3885: Execute Q5_1 GEMV into an existing buffer (zero-allocation, async).
    ///
    /// CONTRACT: `weight_ptr` points at `N * ceil(K/32)` blocks of 24 bytes,
    /// ROW-MAJOR - row `i` starts at `weight_ptr + i * nb * 24`.
    ///
    /// AFFINE: `w = q * d + m`. The per-block min is added after scaling, so a
    /// kernel that drops it gives the right spread around the wrong centre.
    /// Oracle is `quantize::dequantize_q5_1`, verified against llama.cpp's
    /// `dequantize_row_q5_1` line by line in #3869.
    pub fn q5_1_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q5_1_gemv_into")?;
        let kernel_type = KernelType::Q5_1Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5_1_gemv_{}_{}", k, n);
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: pointers validated above; params match the PTX entry signature.
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

    /// #3477: Execute F16 GEMV into an existing buffer (zero-allocation, async).
    ///
    /// CONTRACT: `weight_ptr` points at N*K f16 values, 2 bytes each, ROW-MAJOR —
    /// row `i` starts at `weight_ptr + i * K * 2`. LAYOUT-001: GGUF/APR data is
    /// row-major here and there is no transpose on this path; the `*_colmajor`
    /// kernels are forbidden for it.
    ///
    /// `Qwen3.5-4B-UD-Q4_K_XL` stores `ssm_alpha`/`ssm_beta` as F16. Without this
    /// the hybrid refused the whole model to CPU, because `from_ggml_type(1)`
    /// returned `None` and `hybrid_gpu_unsupported_quant_tensor` named
    /// `blk.0.ssm_alpha` first.
    pub fn f16_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "f16_gemv_into")?;
        let kernel_type = KernelType::F16Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("f16_gemv_{}_{}", k, n);
        // One warp per output row, exactly as every block-quantized GEMV here.
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: pointers validated above; params match the PTX entry signature
        // (y, w, x, k_dim, n_dim).
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

    /// #3908: Execute BF16 GEMV into an existing buffer (zero-allocation, async).
    ///
    /// CONTRACT: `weight_ptr` points at N*K bf16 values, 2 bytes each, ROW-MAJOR
    /// -- row `i` starts at `weight_ptr + i * K * 2`. LAYOUT-001: identical to
    /// [`Self::f16_gemv_into`] above, which is the point: BF16 and F16 are the
    /// same 2-bytes-per-element row-major layout, so the indexing is inherited
    /// rather than re-derived, and the `*_colmajor` kernels stay forbidden.
    ///
    /// `qwen2.5-coder-0.5b-instruct.apr` is 290 BF16 tensors + 1 F32. Before this,
    /// `from_ggml_type(30)` returned `None`, so the .apr CUDA loader declined the
    /// model, wgpu failed on type 30, CPU ran, and the forced-GPU refusal
    /// (R-0b, #3002) correctly reported rc=14 rather than calling it success.
    pub fn bf16_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "bf16_gemv_into")?;
        let kernel_type = KernelType::Bf16Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("bf16_gemv_{}_{}", k, n);
        // One warp per output row, exactly as every block-quantized GEMV here.
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: pointers validated above; params match the PTX entry signature
        // (y, w, x, k_dim, n_dim).
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
