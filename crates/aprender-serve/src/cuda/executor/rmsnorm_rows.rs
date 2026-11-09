impl CudaExecutor {
    /// #5279: [`Self::rmsnorm_into`] over `m` rows: `output[r] = rms_norm(input[r]) * gamma`.
    ///
    /// `input` and `output` are `m × hidden_size`, row-major f32. One launch
    /// covers all rows (row `r` on grid y) with the kernel `rmsnorm_into`
    /// picks on this GPU, so row `r` is bitwise [`Self::rmsnorm_into`] on
    /// `input[r]` alone. The CPU and precise diagnostic modes, and `m == 1`,
    /// run `rmsnorm_into` per row.
    pub fn rmsnorm_rows_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        m: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        static DIAGNOSTIC_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let diagnostic = *DIAGNOSTIC_MODE.get_or_init(|| {
            ["CPU_RMSNORM", "CORRECTNESS_MODE"]
                .iter()
                .any(|var| std::env::var(var).is_ok_and(|v| v == "1"))
        });
        if diagnostic || m <= 1 {
            for r in 0..m {
                let offset = (r * hidden_size) as usize;
                let len = hidden_size as usize;
                // SAFETY: row `r` lies inside both buffers (m rows of
                // `hidden_size`); the views are forgotten, never freed.
                let (row_in, row_out) = unsafe {
                    (
                        GpuBuffer::<f32>::from_raw_parts(input.as_ptr() + offset as u64 * 4, len),
                        GpuBuffer::<f32>::from_raw_parts(output.as_ptr() + offset as u64 * 4, len),
                    )
                };
                let run = self.rmsnorm_into(&row_in, gamma, &row_out, hidden_size, epsilon);
                std::mem::forget(row_in);
                std::mem::forget(row_out);
                run?;
            }
            return Ok(());
        }
        for r in 0..m {
            self.q8_activation_written(output.as_ptr() + u64::from(r * hidden_size) * 4);
        }
        // `rmsnorm_into`'s choice: the single warp on sm_120 and up (GH-559).
        let vectorized = self.gpu_profile.cc < 120;
        let kernel_type = KernelType::RowsRmsNorm {
            hidden_size,
            epsilon,
            vectorized,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!(
            "{kernel_name}_{hidden_size}_{}",
            Self::f32_bits_tag(epsilon)
        );
        self.ensure_kernel_module(&cache_key, &kernel_type)?;
        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");
        let threads = if vectorized { 256 } else { 32 };
        let config = LaunchConfig::grid_2d(1, m, threads, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_gamma = gamma.as_ptr();

        // SAFETY: the buffers hold `m × hidden_size` floats (gamma
        // `hidden_size`) and outlive the stream's run of the kernel.
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // trueno#243: every kernel on a graph path records itself.
        if self.graph_recording {
            let module = self.modules.get_mut(&cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_input, ptr_output, ptr_gamma],
            });
        }

        Ok(())
    }
}
