
impl CudaExecutor {

    /// PAR-058: Execute Q4_1 GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q4_0_gemv_into` but for Q4_1 quantized weights.
    /// Q4_1 adds a min offset (affine quantization) vs Q4_0's symmetric quantization.
    ///
    /// Q4_1 format: 20 bytes per 32 elements (2-byte fp16 scale + 2-byte fp16 min + 16 bytes packed nibbles)
    /// Dequantization: val = d * nibble + m (vs Q4_0's: val = d * (nibble - 8))
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4_1 weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q4_1_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q4_1_gemv_into")?;
        // PAR-058: Zero allocation Q4_1 GEMV for Qwen2.5-0.5B FFN down
        let kernel_type = KernelType::Q4_1Gemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4_1_gemv_{}_{}", k, n);
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

    /// PAR-058: Execute Q5_K GEMV into existing buffer (zero-allocation, async)
    ///
    /// Like `q4k_gemv_into` but for Q5_K quantized weights.
    /// Used when FFN down weights are Q5_K quantized (some GGUF models).
    ///
    /// Q5_K format: 176 bytes per 256 elements
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q5K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q5k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q5k_gemv_into")?;
        // PAR-058: Zero allocation Q5K GEMV for mixed-quantization models
        let kernel_type = KernelType::Q5KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q5k_gemv_{}_{}", k, n);
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
}

include!("weight.rs");
