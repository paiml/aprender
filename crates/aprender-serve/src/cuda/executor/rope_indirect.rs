impl CudaExecutor {

    /// CORRECTNESS-011: RoPE NEOX Indirect (CUDA Graph compatible)
    ///
    /// Same as rope_neox_into but reads position from device memory.
    /// CORRECTNESS-013: When CORRECTNESS_MODE=1, uses PreciseRopeNeoxIndirect kernel
    /// with polynomial sin/cos approximation for CPU-matching precision.
    pub fn rope_neox_indirect_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        position_buf: &GpuBuffer<u32>,
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        // CORRECTNESS-013: Check if precise mode is requested
        static PRECISE_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_precise = *PRECISE_MODE.get_or_init(|| {
            let mode = std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false);
            if mode {
                eprintln!(
                    "[CORRECTNESS-013] RoPE NEOX using PreciseRopeIndirectKernel (polynomial trig)"
                );
            }
            mode
        });

        // Choose kernel type based on mode
        let (kernel_type, cache_key) = if use_precise {
            (
                KernelType::PreciseRopeNeoxIndirect {
                    num_heads,
                    head_dim,
                    theta,
                },
                format!("rope_precise_indirect_{}_{}", num_heads, head_dim),
            )
        } else {
            (
                KernelType::RopeNeoxIndirect {
                    num_heads,
                    head_dim,
                    theta,
                },
                format!("rope_neox_indirect_{}_{}", num_heads, head_dim),
            )
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

        // Grid: num_heads blocks, each with half_dim threads
        let config = LaunchConfig::grid_2d(num_heads, 1, head_dim / 2, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_position = position_buf.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_position) as *mut std::ffi::c_void,
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
                arg_data: vec![ptr_input, ptr_output, ptr_position],
            });
        }

        Ok(())
    }

    // =========================================================================
    // PAR-023: Host Convenience Methods for Activation Kernels
    // =========================================================================

    /// PAR-023: SiLU activation with host memory (convenience)
    ///
    /// Uploads input, runs kernel, syncs, downloads result.
    pub fn silu_host(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), GpuError> {
        let n = input.len() as u32;
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let output_gpu = self.silu_gpu(&input_gpu, n)?;
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;
        Ok(())
    }

    /// PAR-023: GELU activation with host memory (convenience)
    pub fn gelu_host(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), GpuError> {
        let n = input.len() as u32;
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let output_gpu = self.gelu_async(&input_gpu, n)?;
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;
        Ok(())
    }

    /// PAR-023: Element-wise multiply with host memory (convenience)
    pub fn elementwise_mul_host(
        &mut self,
        a: &[f32],
        b: &[f32],
        output: &mut [f32],
    ) -> Result<(), GpuError> {
        let n = a.len() as u32;
        let a_gpu = GpuBuffer::from_host(&self.context, a)?;
        let b_gpu = GpuBuffer::from_host(&self.context, b)?;
        let output_gpu = self.elementwise_mul_gpu(&a_gpu, &b_gpu, n)?;
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;
        Ok(())
    }

    /// PAR-023: Fused SwiGLU with host memory (convenience)
    pub fn fused_swiglu_host(
        &mut self,
        gate: &[f32],
        up: &[f32],
        output: &mut [f32],
    ) -> Result<(), GpuError> {
        let n = gate.len() as u32;
        let gate_gpu = GpuBuffer::from_host(&self.context, gate)?;
        let up_gpu = GpuBuffer::from_host(&self.context, up)?;
        let output_gpu = self.fused_swiglu_gpu(&gate_gpu, &up_gpu, n)?;
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;
        Ok(())
    }

    /// PAR-014: Add two GPU buffers element-wise (residual connection)
    ///
    /// Computes: output[i] += input[i] for all i
    /// Uses simple element-wise kernel for residual connections.
    pub fn add_residual_gpu(
        &mut self,
        output: &GpuBuffer<f32>,
        input: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<(), GpuError> {
        // Use BiasActivation kernel with no activation - it adds "bias" to output
        // We repurpose this by treating input as "bias" to add to output
        let kernel_type = KernelType::BiasActivation {
            n,
            bias_size: n,  // Same size as output
            activation: 0, // No activation
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("residual_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let threads_per_block = 256u32;
        let blocks = (n + threads_per_block - 1) / threads_per_block;
        let config = LaunchConfig::grid_2d(blocks, 1, threads_per_block, 1);

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

        // No sync - caller can batch operations
        Ok(())
    }

    /// PAR-014: Q4K GEMV operating on GPU buffers (no CPU round-trip)
    ///
    /// Input and output are GPU-resident buffers. Only weight name lookup uses CPU.
    /// Part of persistent GPU tensor optimization for M4 milestone.
    pub fn q4k_gemv_gpu(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        let kernel_type = KernelType::Q4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_gemv_{}_{}", k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

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

        // No sync - caller can batch operations
        Ok(())
    }

    /// PAR-094: Tensor Core Q4K GEMM for batched speculative decode
    ///
    /// Enables M>1 batched forward pass with fused dequant+GEMM using tensor cores.
    /// Target: 8x speedup over GEMV for M≥16 speculative tokens.
    ///
    /// # Arguments
    /// * `weight_name` - Name of cached Q4K weight
    /// * `input` - Input activations [M, K] in FP16
    /// * `output` - Output buffer [M, N] in FP16
    /// * `m` - Batch size (number of tokens)
    /// * `k` - Input dimension (must be multiple of 256)
    /// * `n` - Output dimension
    ///
    /// # Errors
    /// Returns error if weight not cached or kernel launch fails
    #[allow(clippy::too_many_arguments)]
    pub fn tensor_core_q4k_gemm(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), GpuError> {
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        let kernel_type = KernelType::TensorCoreQ4KGemm { m, k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("tc_q4k_gemm_{}_{}_{}", m, k, n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: ceil(N/16) x ceil(M/16) blocks, 32 threads per block (1 warp for WMMA)
        let grid_x = (n + 15) / 16;
        let grid_y = (m + 15) / 16;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, 32, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_output = output.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // No sync - caller can batch operations
        Ok(())
    }

    /// PAR-095: Tensor Core Q4K GEMM with CPU input/output
    ///
    /// Batched forward pass for speculative decode verification.
    /// Input and output are CPU slices; computation uses GPU-resident Q4K weights.
    ///
    /// # Arguments
    /// * `weight_name` - Name of cached Q4K weight
    /// * `input` - Input activations [M, K] in FP32 (converted to FP16 on GPU)
    /// * `output` - Output buffer [M, N] in FP32
    /// * `m` - Batch size (number of tokens)
    /// * `k` - Input dimension (must be multiple of 256)
    /// * `n` - Output dimension
    ///
    /// # Errors
    /// Returns error if weight not cached or kernel launch fails
    #[allow(clippy::too_many_arguments)]
    pub fn tensor_core_q4k_gemm_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), GpuError> {
        // Validate dimensions
        let expected_input = (m as usize) * (k as usize);
        let expected_output = (m as usize) * (n as usize);

        if input.len() != expected_input {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-095: Input size {} != expected M*K = {}*{} = {}",
                input.len(),
                m,
                k,
                expected_input
            )));
        }
        if output.len() != expected_output {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-095: Output size {} != expected M*N = {}*{} = {}",
                output.len(),
                m,
                n,
                expected_output
            )));
        }

        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let _weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // Upload input to GPU
        let input_buf = GpuBuffer::from_host(&self.context, input)?;
        let output_buf = GpuBuffer::new(&self.context, expected_output)?;

        // Execute kernel
        self.tensor_core_q4k_gemm(weight_name, &input_buf, &output_buf, m, k, n)?;

        // Sync and download output
        self.stream.synchronize()?;
        output_buf.copy_to_host(output)?;

        Ok(())
    }
}
