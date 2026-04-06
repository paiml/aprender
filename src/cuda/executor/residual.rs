
impl CudaExecutor {

    /// PAR-114: Batched SwiGLU kernel for M sequences
    ///
    /// Fused SiLU+multiply for M sequences in parallel.
    /// Reduces M kernel launches to 1.
    ///
    /// # Arguments
    ///
    /// * `gate` - Packed gate values [M × n]
    /// * `up` - Packed up values [M × n]
    /// * `output` - Output [M × n]
    /// * `n` - Elements per sequence
    /// * `batch_size` - Number of sequences (M)
    pub fn batched_swiglu_into(
        &mut self,
        gate: &GpuBuffer<f32>,
        up: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        batch_size: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::BatchedSwiglu { n, batch_size };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        // GH-129: PTX depends on n (immediate) but NOT batch_size (grid dim).
        let cache_key = format!("batched_swiglu_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-114: Grid (ceil(n/256), batch_size, 1) with 256 threads
        let blocks_x = (n + 255) / 256;
        let config = LaunchConfig::grid_2d(blocks_x, batch_size, 256, 1);

        let mut ptr_gate = gate.as_ptr();
        let mut ptr_up = up.as_ptr();
        let mut ptr_output = output.as_ptr();

        // SAFETY: Pointers derived from valid GpuBuffer refs, kernel config matches data dimensions
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_gate) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-023: RMSNorm on GPU with host input/output (synchronous convenience method)
    ///
    /// This is a convenience wrapper around `rmsnorm_gpu` that handles
    /// host-to-device and device-to-host transfers.
    ///
    /// # Arguments
    ///
    /// * `input` - Host slice with input vector [hidden_size]
    /// * `gamma` - Host slice with scale weights [hidden_size]
    /// * `output` - Host slice for output [hidden_size]
    /// * `epsilon` - Numerical stability constant (default: 1e-5)
    pub fn rmsnorm_host(
        &mut self,
        input: &[f32],
        gamma: &[f32],
        output: &mut [f32],
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let hidden_size = input.len() as u32;

        // Upload to GPU
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let gamma_gpu = GpuBuffer::from_host(&self.context, gamma)?;

        // Run kernel
        let output_gpu = self.rmsnorm_gpu(&input_gpu, &gamma_gpu, hidden_size, epsilon)?;

        // Sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-023: Residual Add on GPU with host input/output (synchronous convenience method)
    ///
    /// This is a convenience wrapper around `residual_add_gpu` that handles
    /// host-to-device and device-to-host transfers.
    ///
    /// # Arguments
    ///
    /// * `input1` - Host slice with first input vector
    /// * `input2` - Host slice with second input vector
    /// * `output` - Host slice for output
    pub fn residual_add_host(
        &mut self,
        input1: &[f32],
        input2: &[f32],
        output: &mut [f32],
    ) -> Result<(), GpuError> {
        let n = input1.len() as u32;

        // Upload to GPU
        let input1_gpu = GpuBuffer::from_host(&self.context, input1)?;
        let input2_gpu = GpuBuffer::from_host(&self.context, input2)?;

        // Run kernel
        let output_gpu = self.residual_add_gpu(&input1_gpu, &input2_gpu, n)?;

        // Sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-023: Fused Residual Add + RMSNorm with host input/output (synchronous convenience method)
    ///
    /// This is a convenience wrapper around `fused_residual_rmsnorm_gpu` that handles
    /// host-to-device and device-to-host transfers.
    ///
    /// # Arguments
    ///
    /// * `residual` - Host slice with residual input
    /// * `input` - Host slice with input to add
    /// * `gamma` - Host slice with scale weights
    /// * `output` - Host slice for output
    /// * `epsilon` - Numerical stability constant
    pub fn fused_residual_rmsnorm_host(
        &mut self,
        residual: &[f32],
        input: &[f32],
        gamma: &[f32],
        output: &mut [f32],
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let hidden_size = residual.len() as u32;

        // Upload to GPU
        let residual_gpu = GpuBuffer::from_host(&self.context, residual)?;
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;
        let gamma_gpu = GpuBuffer::from_host(&self.context, gamma)?;

        // Run kernel
        let output_gpu = self.fused_residual_rmsnorm_gpu(
            &residual_gpu,
            &input_gpu,
            &gamma_gpu,
            hidden_size,
            epsilon,
        )?;

        // Sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-023: Residual Add using dedicated kernel (async)
    ///
    /// Computes: output[i] = input1[i] + input2[i]
    /// Uses the new ResidualAddKernel for better async pipeline integration.
    ///
    /// # Arguments
    ///
    /// * `input1` - First input buffer
    /// * `input2` - Second input buffer
    /// * `n` - Number of elements
    ///
    /// # Returns
    ///
    /// GPU buffer with result (no sync - async)
    pub fn residual_add_gpu(
        &mut self,
        input1: &GpuBuffer<f32>,
        input2: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::ResidualAdd { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        // GH-129: PTX is n-independent (n is a runtime param), so use constant cache key.
        // Prevents JIT recompilation for different n values on memory-constrained devices.
        let cache_key = "residual_add".to_string();

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
        let output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // 256 threads per block
        let threads_per_block = 256u32;
        let blocks = (n + threads_per_block - 1) / threads_per_block;
        let config = LaunchConfig::grid_2d(blocks, 1, threads_per_block, 1);

        let mut ptr_input1 = input1.as_ptr();
        let mut ptr_input2 = input2.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input1) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input2) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-023: NO sync - async operation for pipeline
        Ok(output)
    }

    /// PAR-044: Residual add into existing buffer (zero-allocation, async)
    #[inline]
    pub fn residual_add_into(
        &mut self,
        input1: &GpuBuffer<f32>,
        input2: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::ResidualAdd { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        // GH-129: PTX is n-independent (n is a runtime param), so use constant cache key.
        let cache_key = "residual_add".to_string();

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

        let mut ptr_input1 = input1.as_ptr();
        let mut ptr_input2 = input2.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input1) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input2) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
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
                arg_data: vec![ptr_input1, ptr_input2, ptr_output, n_val as u64],
            });
        }

        Ok(())
    }

    /// PAR-023: Fused Residual Add + RMSNorm (async)
    ///
    /// Computes: output = rmsnorm(residual + input, gamma, epsilon)
    /// Fuses residual add and normalization to reduce memory bandwidth.
    ///
    /// # Arguments
    ///
    /// * `residual` - Residual input buffer
    /// * `input` - Input to add to residual
    /// * `gamma` - RMSNorm scale weights
    /// * `hidden_size` - Hidden dimension
    /// * `epsilon` - Numerical stability constant
    ///
    /// # Returns
    ///
    /// GPU buffer with normalized result (no sync - async)
    pub fn fused_residual_rmsnorm_gpu(
        &mut self,
        residual: &GpuBuffer<f32>,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        hidden_size: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::FusedResidualRmsNorm {
            hidden_size,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_residual_rmsnorm_{}", hidden_size);

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
        let output = GpuBuffer::<f32>::new(&self.context, hidden_size as usize)?;

        // Fused kernel uses one warp (32 threads)
        let config = LaunchConfig::grid_2d(1, 1, 32, 1);

        let mut ptr_residual = residual.as_ptr();
        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_gamma = gamma.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_residual) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-023: NO sync - async operation for pipeline
        Ok(output)
    }

    /// PAR-075: Fused Residual Add + RMSNorm into pre-allocated buffer
    ///
    /// Computes: output = rmsnorm(residual + input, gamma, epsilon)
    /// Fuses residual add and normalization to reduce memory bandwidth.
    /// Uses pre-allocated output buffer to eliminate allocation.
    ///
    /// NOTE: input == output is safe for this kernel due to:
    /// 1. Single-warp execution (lockstep within warp)
    /// 2. Each thread handles disjoint elements
    /// 3. Read before write per element per thread
    pub fn fused_residual_rmsnorm_into(
        &mut self,
        residual: &GpuBuffer<f32>,
        input: &GpuBuffer<f32>,
        gamma_ptr: usize, // Raw device pointer to gamma weights
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedResidualRmsNorm {
            hidden_size,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_residual_rmsnorm_{}", hidden_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Fused kernel uses one warp (32 threads)
        let config = LaunchConfig::grid_2d(1, 1, 32, 1);

        let mut ptr_residual = residual.as_ptr();
        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_gamma = gamma_ptr;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_residual) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-075: NO sync - async operation for pipeline
        Ok(())
    }
}

include!("tiled_gemv.rs");
include!("q4k_q8_gemv.rs");
include!("layer_norm_gpu.rs");
