impl CudaExecutor {

    /// PAR-082-V2: Multi-warp Vectorized Q4K GEMV (4 warps + u32 coalesced loads)
    ///
    /// Combines VectorizedQ4K's coalesced u32 loads with multi-warp parallelism.
    /// 128 threads (4 warps) per block, matching llama.cpp's mul_mat_vec_q.
    pub fn mwv_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "mwv_q4k_gemv_into")?;
        // PAR-082-V3: Configurable warp count via MWV_WARPS env var (default: 2)
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

        // num_warps * 32 threads per output element, one block per output
        let threads = num_warps * 32;
        let config = LaunchConfig::grid_2d(n, 1, threads, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All device pointers (output, weights, input) are allocated by
        // CudaExecutor and valid for the kernel's grid dimensions. k_val and n_val
        // are stack-local scalars passed by pointer. The stream synchronizes before
        // any host-side read of the output buffer.
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

    /// PAR-082-V4: Execute MWV DP4A Q4_K GEMV with Q8_1-quantized activations
    ///
    /// Two-step pipeline:
    /// 1. Quantize f32 activations → Q8_1 format (Q8QuantizeKernel)
    /// 2. DP4A dot product: Q4K weights × Q8_1 activations → f32 output
    ///
    /// 3.3x instruction reduction vs MWV, coalesced Q8 loads vs scattered f32.
    /// Enable with `DP4A_Q4K=1` env var.
    pub fn mwv_dp4a_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "mwv_dp4a_q4k_gemv_into")?;

        // PAR-PERF-DP4A: Use pre-allocated Q8 buffer from workspace (zero allocation)
        // Five-Whys root cause: Old code did GpuBuffer::new per call = 280 cudaMallocs/token
        // Fix: Borrow workspace.q8_activation_buf, quantize in-place, then launch GEMV
        let q8_ptr = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("PAR-PERF-DP4A: workspace.q8_activation_buf not initialized. Call init_workspace() first.")
            .as_ptr();
        let q8_len = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("q8_activation_buf must be initialized")
            .len();

        // Create non-owning view into the pre-allocated Q8 buffer
        // SAFETY: q8_activation_buf is pre-allocated in init_workspace and valid for this scope
        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };

        // Step 1: Quantize activations to Q8_1 (skip if already valid — PMAT-027)
        if !self.q8_activation_valid {
            self.q8_quantize_into(input, &q8_buf, k)?;
            self.q8_activation_valid = true;
        }

        // Step 2: Launch DP4A GEMV kernel
        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::MwvDp4aQ4KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("mwv_dp4a_q4k_gemv_{}_{}_{}", k, n, num_warps);

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
        // Scale grid to fill SM warp slots: 48 warps/SM ÷ 3 warps/block = 16 blocks/SM.
        // Jetson (8 SMs) → 128 blocks, 4090 (128 SMs) → 2048 blocks.
        let grid_x = n.min(self.num_sms * 16);
        let config = LaunchConfig::grid_2d(grid_x, 1, threads, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_q8 = q8_buf.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All device pointers (output, weights, q8_buf) are allocated by
        // CudaExecutor and valid for the kernel's grid dimensions. k_val and n_val
        // are stack-local scalars passed by pointer. q8_buf is a view into
        // workspace.q8_activation_buf and is kept alive via std::mem::forget below.
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
        // BUG FIX (realizr#198): MWV DP4A variant was missing recording.
        if self.graph_recording {
            let module = self.modules.get_mut(&cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_output, ptr_weights, ptr_q8, k_val as u64, n_val as u64],
            });
        }

        // Don't drop — q8_buf is a view into workspace.q8_activation_buf
        std::mem::forget(q8_buf);

        Ok(())
    }

    /// GH-176: Half-warp DP4A Q4_K GEMV (16 threads/SB, 1.77x fewer thread-insn)
    ///
    /// Same two-step pipeline as MWV DP4A but uses half-warp kernel:
    /// 1. Quantize f32 activations → Q8_1 format
    /// 2. Half-warp DP4A dot product: Q4K × Q8_1 → f32
    ///
    /// Enable with `HW_DP4A_Q4K=1` env var.
    pub fn hw_dp4a_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "hw_dp4a_q4k_gemv_into")?;

        let q8_ptr = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("hw_dp4a: workspace.q8_activation_buf not initialized")
            .as_ptr();
        let q8_len = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("q8_activation_buf must be initialized")
            .len();

        // SAFETY: q8_activation_buf is pre-allocated in init_workspace and valid for this scope
        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };

        // Step 1: Quantize activations to Q8_1 (skip if already valid — PMAT-027)
        if !self.q8_activation_valid {
            self.q8_quantize_into(input, &q8_buf, k)?;
            self.q8_activation_valid = true;
        }

        // Step 2: Launch half-warp DP4A GEMV kernel
        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::HwDp4aQ4KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("hw_dp4a_q4k_gemv_{}_{}_{}", k, n, num_warps);

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

        // SAFETY: All device pointers are allocated by CudaExecutor and valid.
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

        // trueno#243: Record kernel AFTER launch (avoids borrow conflict with module)
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

    /// PAR-063: Execute DP4A Q4_K GEMV into existing buffer
    ///
    /// Uses DP4A SIMD instruction for 4x instruction reduction.
    /// Each DP4A computes 4 multiply-adds in a single instruction.
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be multiple of 256)
    #[inline]
    pub fn dp4a_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "dp4a_q4k_gemv_into")?;
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

    /// PAR-076: Execute Fused RMSNorm + Q4_K GEMV into existing buffer
    ///
    /// Fuses RMSNorm normalization with Q4K GEMV in a single kernel pass:
    /// output = matmul(weights, rmsnorm(input, gamma))
    ///
    /// Phase 1: Load input, compute sum of squares, normalize in shared memory
    /// Phase 2: Do Q4K GEMV using normalized values from shared memory
    ///
    /// Eliminates:
    /// - Separate RMSNorm kernel launch (~1.5µs)
    /// - Global memory round-trip for normalized values
    /// - Memory bandwidth for writing/reading normalized buffer
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector (K elements, NOT normalized)
    /// * `gamma_ptr` - RMSNorm scale weights (K elements)
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `k` - Input/hidden dimension (must be multiple of 256)
    /// * `n` - Output dimension
    /// * `epsilon` - RMSNorm numerical stability (default 1e-5)
    #[inline]
    pub fn fused_rmsnorm_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        gamma_ptr: u64,
        output: &GpuBuffer<f32>,
        k: u32,
        n: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "fused_rmsnorm_q4k_gemv_into(weight)")?;
        validate_device_ptr(gamma_ptr, "fused_rmsnorm_q4k_gemv_into(gamma)")?;
        let kernel_type = KernelType::FusedRmsNormQ4KGemv { k, n, epsilon };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_rmsnorm_q4k_gemv_{}_{}_{:.0e}", k, n, epsilon);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // One block per output element, 256 threads per block
        let config = LaunchConfig::grid_2d(n, 1, 256, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut ptr_gamma = gamma_ptr;
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
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-034: Execute Fused Gate+Up+SwiGLU HW DP4A Q4_K GEMV
    ///
    /// Computes `result[i] = silu(dot(W_gate[i], x)) * dot(W_up[i], x)` in a single
    /// kernel pass. Eliminates gate_out/up_out intermediate buffers and SwiGLU kernel.
    ///
    /// Uses Q8-quantized activations (shared between gate and up dot products).
    #[inline]
    pub fn fused_gate_up_swiglu_hw_dp4a_q4k_gemv_into(
        &mut self,
        gate_weight_ptr: u64,
        up_weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        k: u32,
        n: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(gate_weight_ptr, "fused_gate_up_swiglu(gate)")?;
        validate_device_ptr(up_weight_ptr, "fused_gate_up_swiglu(up)")?;

        // Q8 quantize activations (skip if already valid — PMAT-027 cache)
        let q8_ptr = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("q8_activation_buf must be initialized")
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
        let kernel_type = KernelType::FusedGateUpSwigluHwDp4aQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_gate_up_swiglu_hw_dp4a_q4k_{}_{}", k, n);

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
        let mut ptr_gate_weights = gate_weight_ptr;
        let mut ptr_up_weights = up_weight_ptr;
        let mut ptr_q8 = q8_buf.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All device pointers are allocated by CudaExecutor and valid.
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gate_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_q8) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // trueno#243: Record kernel for manual graph construction
        // BUG FIX (realizr#198): This recording was MISSING — 28 fused gate+up+SwiGLU
        // kernels per forward pass were never captured. Graph replay used stale
        // ffn_act_buf from the eager pass, causing hidden_buf2 divergence (81.0).
        if self.graph_recording {
            let module = self.modules.get_mut(&cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_output, ptr_gate_weights, ptr_up_weights, ptr_q8, k_val as u64, n_val as u64],
            });
        }

        std::mem::forget(q8_buf);

        Ok(())
    }

    /// GH-288: Fused Q+K+V HW DP4A Q4_K GEMV
    ///
    /// Computes all three attention projections, sharing Q8-quantized input:
    ///   Q[q_n] = W_q[q_n, k] × x[k]
    ///   K[kv_n] = W_k[kv_n, k] × x[k]
    ///   V[kv_n] = W_v[kv_n, k] × x[k]
    ///
    /// Phase 1 (current): Shared Q8 quantization, 3 GEMV launches (saves Q8 recompute).
    /// Phase 2 (trueno#237): Single fused kernel launch (saves 2 more launches).
    ///
    /// GQA-aware: q_n may differ from kv_n.
    /// Phase 2 (trueno#237): Q stays separate, K+V use fused kernel (1 launch instead of 2).
    #[inline]
    pub fn fused_qkv_hw_dp4a_q4k_gemv_into(
        &mut self,
        q_weight_ptr: u64,
        k_weight_ptr: u64,
        v_weight_ptr: u64,
        input: &GpuBuffer<f32>,
        q_output: &GpuBuffer<f32>,
        k_output: &GpuBuffer<f32>,
        v_output: &GpuBuffer<f32>,
        k_dim: u32,
        q_n: u32,
        kv_n: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(q_weight_ptr, "fused_qkv(q)")?;
        validate_device_ptr(k_weight_ptr, "fused_qkv(k)")?;
        validate_device_ptr(v_weight_ptr, "fused_qkv(v)")?;

        // GH-288 Phase 1: Q8 quantization shared via PMAT-027 cache.
        // Phase 2 (trueno#237 fused K+V kernel) MEASURED: -3.1% regression at
        // kv_dim=256 (Qwen2.5-1.5B). Launch overhead savings (~5μs) smaller than
        // compute overhead from dual accumulators+weight loads (~10μs/row).
        // Fused kernel left available for future models with larger kv_dim.
        self.q4k_gemv_into(q_weight_ptr, input, q_output, q_n, k_dim)?;
        // q8_activation_valid is now true — K and V skip quantization
        self.q4k_gemv_into(k_weight_ptr, input, k_output, kv_n, k_dim)?;
        self.q4k_gemv_into(v_weight_ptr, input, v_output, kv_n, k_dim)?;

        Ok(())
    }

    /// trueno#237: Fused K+V HW DP4A Q4K GEMV — 2 projections in 1 kernel launch.
    ///
    /// Uses Q8-quantized activations (shared between K and V dot products).
    /// Same pattern as `fused_gate_up_swiglu_hw_dp4a_q4k_gemv_into` but with
    /// two raw outputs instead of SwiGLU activation.
    fn fused_kv_hw_dp4a_q4k_gemv_into(
        &mut self,
        k_weight_ptr: u64,
        v_weight_ptr: u64,
        input: &GpuBuffer<f32>,
        k_output: &GpuBuffer<f32>,
        v_output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Q8 quantize activations (skip if already valid — PMAT-027 cache)
        let q8_ptr = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("fused_kv: workspace.q8_activation_buf not initialized")
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
        let kernel_type = KernelType::FusedKVHwDp4aQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_kv_hw_dp4a_q4k_{}_{}", k, n);

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

        let mut ptr_q8 = q8_buf.as_ptr();
        let mut ptr_k_weights = k_weight_ptr;
        let mut ptr_v_weights = v_weight_ptr;
        let mut ptr_k_output = k_output.as_ptr();
        let mut ptr_v_output = v_output.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All device pointers are allocated by CudaExecutor and valid.
        // Kernel params match trueno FusedQKVHwDp4aQ4KGemvKernel: x_ptr, wk_ptr, wv_ptr, y_k_ptr, y_v_ptr, k_dim, n_dim
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_q8) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_k_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_v_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_k_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_v_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        std::mem::forget(q8_buf);

        Ok(())
    }

    /// GH-141: Batched HW DP4A Q4_K GEMV for M=2..8
    ///
    /// Three-step pipeline:
    /// 1. Quantize M×K f32 activations → M×Q8_1 blocks (one kernel launch)
    /// 2. Launch batched HW DP4A kernel: Q4K weights × M Q8_1 activations
    /// 3. Output: M×N f32 (row-major)
    ///
    /// 4.7x less bandwidth than cuBLAS SGEMM (Q4K 0.5625 + Q8_1 1.125 vs FP32 8 B/elem).
    pub fn batched_hw_dp4a_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "batched_hw_dp4a_q4k_gemv_into")?;

        // Q8_1 buffer: M vectors × (K/32 blocks × 36 bytes)
        let q8_blocks_per_vec = (k + 31) / 32;
        let q8_bytes_per_vec = q8_blocks_per_vec * 36;
        let q8_total_bytes = (m * q8_bytes_per_vec) as usize;

        // Use workspace q8_activation_buf if large enough, else allocate
        let q8_ptr = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("batched_hw_dp4a: workspace.q8_activation_buf not initialized")
            .as_ptr();
        let q8_len = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("q8_activation_buf must be initialized")
            .len();

        if q8_len < q8_total_bytes {
            return Err(GpuError::InvalidParameter(format!(
                "GH-141: q8_activation_buf too small: have {} bytes, need {} for M={m}",
                q8_len, q8_total_bytes
            )));
        }

        // SAFETY: q8_activation_buf is pre-allocated in init_workspace and valid for this scope
        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };

        // PMAT-294: Check Q8 activation cache before quantizing.
        // Skip Q8 quantize when the same input buffer was already quantized
        // (e.g., K/V projections share input with Q, up shares with gate).
        // Saves 3 Q8 launches per layer × 28 layers = 84 launches per step.
        if !self.q8_activation_valid {
            let total_elements = m * k;
            self.q8_quantize_into(input, &q8_buf, total_elements)?;
            self.q8_activation_valid = true;
        }

        // Step 2: Launch batched HW DP4A kernel
        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::BatchedHwDp4aQ4KGemv { k, n, m, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("batched_hw_dp4a_q4k_gemv_{}_{}_{}_{}", k, n, m, num_warps);

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
        let mut m_val = m;

        // SAFETY: All device pointers are allocated by CudaExecutor and valid.
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
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        std::mem::forget(q8_buf);

        Ok(())
    }

    /// GH-141: Launch batched HW DP4A kernel with pre-quantized Q8_1 activations.
    ///
    /// Skips Q8_1 quantization — caller must ensure `q8_ptr` contains valid Q8_1 data
    /// for M vectors of K dimensions. Used for gate+up fusion (quantize once, launch twice).
    fn batched_hw_dp4a_q4k_gemv_q8_launch(
        &mut self,
        weight_ptr: u64,
        q8_ptr: u64,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::BatchedHwDp4aQ4KGemv { k, n, m, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("batched_hw_dp4a_q4k_gemv_{}_{}_{}_{}", k, n, m, num_warps);

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
        let mut ptr_q8 = q8_ptr;
        let mut k_val = k;
        let mut n_val = n;
        let mut m_val = m;

        // SAFETY: All device pointers are allocated by CudaExecutor and valid.
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
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-295: Inline Q8 DP4A Q4K GEMV — fuses Q8 quantize into GEMV.
    ///
    /// Single kernel: reads FP32 input, quantizes to INT8 per-thread, DP4A with Q4K.
    /// No separate Q8 quantize launch needed.
    pub fn inline_q8_dp4a_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "inline_q8_dp4a_q4k_gemv_into")?;

        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::InlineQ8Dp4aQ4KGemv { k, n, m, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("inline_q8_dp4a_q4k_gemv_{}_{}_{}_{}", k, n, m, num_warps);

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
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;
        let mut m_val = m;

        // SAFETY: All device pointers are allocated by CudaExecutor and valid.
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

    /// PMAT-293: Fused FP32-input Q4K GEMV (no Q8 pre-quantize).
    ///
    /// Single kernel launch: reads FP32 activations + Q4K weights, dequants to FP32,
    /// FMA accumulate. Eliminates the separate Q8_1 quantize kernel launch.
    pub fn fused_fp32_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "fused_fp32_q4k_gemv_into")?;

        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::FusedFp32Q4KGemv { k, n, m, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_fp32_q4k_gemv_{}_{}_{}_{}", k, n, m, num_warps);

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
        let mut ptr_input = input.as_ptr(); // FP32 input directly (NOT Q8!)
        let mut k_val = k;
        let mut n_val = n;
        let mut m_val = m;

        // SAFETY: All device pointers are allocated by CudaExecutor and valid.
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

    /// GH-141: Batched gate+up DP4A GEMV with shared Q8_1 quantization.
    ///
    /// Quantizes the input ONCE, then launches both gate and up GEMV kernels
    /// using the same Q8_1 buffer. Saves one Q8 quantize kernel per layer (28 total).
    #[allow(clippy::too_many_arguments)]
    pub fn batched_gate_up_dp4a_q4k_gemv_into(
        &mut self,
        gate_weight_ptr: u64,
        up_weight_ptr: u64,
        input: &GpuBuffer<f32>,
        gate_output: &GpuBuffer<f32>,
        up_output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(gate_weight_ptr, "batched_gate_up_dp4a: gate")?;
        validate_device_ptr(up_weight_ptr, "batched_gate_up_dp4a: up")?;

        // Q8_1 buffer: M vectors × (K/32 blocks × 36 bytes)
        let q8_blocks_per_vec = (k + 31) / 32;
        let q8_bytes_per_vec = q8_blocks_per_vec * 36;
        let q8_total_bytes = (m * q8_bytes_per_vec) as usize;

        let q8_ptr = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("batched_gate_up_dp4a: q8_activation_buf not initialized")
            .as_ptr();
        let q8_len = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("q8_activation_buf must be initialized")
            .len();

        if q8_len < q8_total_bytes {
            return Err(GpuError::InvalidParameter(format!(
                "GH-141: q8_activation_buf too small for gate+up: have {} bytes, need {} for M={m}",
                q8_len, q8_total_bytes
            )));
        }

        // SAFETY: q8_activation_buf is pre-allocated and valid for this scope
        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };

        // Step 1: Quantize input ONCE (shared across gate + up)
        let total_elements = m * k;
        self.q8_quantize_into(input, &q8_buf, total_elements)?;

        // Step 2: Launch gate GEMV using pre-quantized Q8_1
        self.batched_hw_dp4a_q4k_gemv_q8_launch(
            gate_weight_ptr, q8_ptr, gate_output, m, n, k,
        )?;

        // Step 3: Launch up GEMV using same Q8_1 buffer
        self.batched_hw_dp4a_q4k_gemv_q8_launch(
            up_weight_ptr, q8_ptr, up_output, m, n, k,
        )?;

        std::mem::forget(q8_buf);
        Ok(())
    }

    /// PMAT-054A: Fused QKV DP4A GEMV with shared Q8_1 quantization.
    ///
    /// Quantizes the input ONCE, then launches Q, K, V GEMV kernels
    /// using the same Q8_1 buffer. Saves 2 Q8 quantize kernels per layer
    /// (56 total across 28 layers).
    ///
    /// Q projection has output dim `q_dim`, K and V have output dim `kv_dim`.
    /// All share the same input dim `k` (hidden_dim after RMSNorm).
    #[allow(clippy::too_many_arguments)]
    pub fn batched_qkv_dp4a_q4k_gemv_into(
        &mut self,
        q_weight_ptr: u64,
        k_weight_ptr: u64,
        v_weight_ptr: u64,
        input: &GpuBuffer<f32>,
        q_output: &GpuBuffer<f32>,
        k_output: &GpuBuffer<f32>,
        v_output: &GpuBuffer<f32>,
        m: u32,
        q_dim: u32,
        kv_dim: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(q_weight_ptr, "batched_qkv_dp4a: Q")?;
        validate_device_ptr(k_weight_ptr, "batched_qkv_dp4a: K")?;
        validate_device_ptr(v_weight_ptr, "batched_qkv_dp4a: V")?;

        // Q8_1 buffer: M vectors × (K/32 blocks × 36 bytes)
        let q8_blocks_per_vec = (k + 31) / 32;
        let q8_bytes_per_vec = q8_blocks_per_vec * 36;
        let q8_total_bytes = (m * q8_bytes_per_vec) as usize;

        let q8_ptr = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("batched_qkv_dp4a: q8_activation_buf not initialized")
            .as_ptr();
        let q8_len = self
            .workspace
            .q8_activation_buf
            .as_ref()
            .expect("q8_activation_buf must be initialized")
            .len();

        if q8_len < q8_total_bytes {
            return Err(GpuError::InvalidParameter(format!(
                "PMAT-054A: q8_activation_buf too small for QKV: have {} bytes, need {} for M={m}",
                q8_len, q8_total_bytes
            )));
        }

        // SAFETY: q8_activation_buf is pre-allocated and valid for this scope
        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };

        // Step 1: Quantize input ONCE (shared across Q + K + V)
        let total_elements = m * k;
        self.q8_quantize_into(input, &q8_buf, total_elements)?;

        // Step 2: Launch Q GEMV using pre-quantized Q8_1
        self.batched_hw_dp4a_q4k_gemv_q8_launch(
            q_weight_ptr, q8_ptr, q_output, m, q_dim, k,
        )?;

        // Step 3: Launch K GEMV using same Q8_1 buffer
        self.batched_hw_dp4a_q4k_gemv_q8_launch(
            k_weight_ptr, q8_ptr, k_output, m, kv_dim, k,
        )?;

        // Step 4: Launch V GEMV using same Q8_1 buffer
        self.batched_hw_dp4a_q4k_gemv_q8_launch(
            v_weight_ptr, q8_ptr, v_output, m, kv_dim, k,
        )?;

        std::mem::forget(q8_buf);
        Ok(())
    }

    /// PAR-077: Execute Fused Gate+Up Q4_K GEMV into existing buffers
    ///
    /// Computes both gate and up projections in a single kernel pass:
    ///   gate_out = W_gate * x
    ///   up_out = W_up * x
    ///
    /// Optimization: Reads input x only ONCE (saved to shared memory)
    /// - Standard approach: 2 kernel launches, 2x input bandwidth
    /// - Fused approach: 1 kernel launch, 1x input bandwidth
    ///
    /// Expected savings: ~30% reduction in FFNGateUp time
    ///
    /// # Arguments
    ///
    /// * `gate_weight_ptr` - Raw device pointer to Q4K gate weights
    /// * `up_weight_ptr` - Raw device pointer to Q4K up weights
    /// * `input` - GPU buffer containing input vector (K elements)
    /// * `gate_output` - Pre-allocated output buffer for gate (N elements)
    /// * `up_output` - Pre-allocated output buffer for up (N elements)
    /// * `k` - Input/hidden dimension (must be multiple of 256)
    /// * `n` - Intermediate dimension (output size)
    #[inline]
    pub fn fused_gate_up_q4k_gemv_into(
        &mut self,
        gate_weight_ptr: u64,
        up_weight_ptr: u64,
        input: &GpuBuffer<f32>,
        gate_output: &GpuBuffer<f32>,
        up_output: &GpuBuffer<f32>,
        k: u32,
        n: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(gate_weight_ptr, "fused_gate_up_q4k_gemv_into(gate)")?;
        validate_device_ptr(up_weight_ptr, "fused_gate_up_q4k_gemv_into(up)")?;
        let kernel_type = KernelType::FusedGateUpQ4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_gate_up_q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // One block per output element, 256 threads per block
        let config = LaunchConfig::grid_2d(n, 1, 256, 1);

        let mut ptr_gate_out = gate_output.as_ptr();
        let mut ptr_up_out = up_output.as_ptr();
        let mut ptr_gate_weights = gate_weight_ptr;
        let mut ptr_up_weights = up_weight_ptr;
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
                    std::ptr::from_mut(&mut ptr_gate_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gate_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }
}
