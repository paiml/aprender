impl CudaExecutor {
    /// #5279: Q5_K GEMV over `m` activation rows: `output[r] = W · input[r]`.
    ///
    /// `input` is `m × k` and `output` is `m × n`, both row-major f32. Each
    /// weight value is dequantized once for all `m` rows, and row `r` of
    /// `output` is bitwise equal to [`Self::q5k_gemv_into`] on `input[r]`
    /// alone.
    pub fn batched_q5k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "batched_q5k_gemv_into")?;
        debug_assert!(
            input.len() >= (m * k) as usize && output.len() >= (m * n) as usize,
            "batched_q5k_gemv_into: buffers hold fewer than m rows"
        );
        let kernel_type = KernelType::BatchedQ5KGemv { k, n, m };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = module_key!(self, "batched_q5k_gemv_{}_{}_{}", k, n, m);
        // As the single-row kernel: one warp per output.
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&*cache_key)
            .expect("module just inserted");

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All device pointers (output, weights, input) are allocated by
        // CudaExecutor; input holds m × k and output m × n f32 values, which is
        // all the kernel touches. k_val and n_val are stack-local scalars passed
        // by pointer. The stream synchronizes before any host-side read of the
        // output buffer.
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

        // Record for manual graph construction, as q5k_gemv_into does: a
        // launch missing from the graph leaves its output stale on replay.
        if self.graph_recording {
            let module = self.modules.get_mut(&*cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![
                    ptr_output,
                    ptr_weights,
                    ptr_input,
                    k_val as u64,
                    n_val as u64,
                ],
            });
        }

        Ok(())
    }
}
