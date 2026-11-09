impl CudaExecutor {
    /// #5279: HW DP4A Q6_K GEMV over `m` activation rows:
    /// `output[r] = W · input[r]`, `k` a multiple of 256.
    ///
    /// `input` is `m × k` and `output` is `m × n`, both row-major f32. All `m`
    /// rows are quantized to Q8_1 in one launch (a Q8_1 block never spans two
    /// rows, so each row's bytes are the ones a single-row quantize gives),
    /// then each weight super-block is read once for all rows. Row `r` of
    /// `output` is bitwise equal to [`Self::hw_dp4a_q6k_gemv_into`] on
    /// `input[r]` alone.
    ///
    /// The Q8 rows go to `workspace.batched_q8_buf`, never to
    /// `workspace.q8_activation_buf`, so the single-row Q8 cache stays valid.
    pub fn batched_hw_dp4a_q6k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "batched_hw_dp4a_q6k_gemv_into")?;
        debug_assert!(
            k.is_multiple_of(256),
            "batched_hw_dp4a_q6k_gemv_into: k must be a multiple of 256"
        );
        debug_assert!(
            input.len() >= (m * k) as usize && output.len() >= (m * n) as usize,
            "batched_hw_dp4a_q6k_gemv_into: buffers hold fewer than m rows"
        );

        // m rows of k / 32 Q8_1 blocks, 36 bytes each.
        let q8_bytes = (m * (k / 32) * 36) as usize;
        let have = self
            .workspace
            .batched_q8_buf
            .as_ref()
            .map_or(0, GpuBuffer::len);
        if have < q8_bytes {
            self.workspace.batched_q8_buf = Some(GpuBuffer::new(&self.context, q8_bytes)?);
        }
        let q8 = self
            .workspace
            .batched_q8_buf
            .as_ref()
            .expect("batched_q8_buf just sized");
        let (q8_ptr, q8_len) = (q8.as_ptr(), q8.len());
        // SAFETY: a non-owning view of `workspace.batched_q8_buf`, which stays
        // allocated for this call; it is forgotten below, so its Drop never
        // frees the workspace's allocation.
        let q8_buf = unsafe { GpuBuffer::<u8>::from_raw_parts(q8_ptr, q8_len) };
        let quantized = self.q8_quantize_into(input, &q8_buf, m * k);
        std::mem::forget(q8_buf);
        quantized?;

        // The same warp count as the single-row launch: the per-row sum order
        // depends on it.
        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::BatchedHwDp4aQ6KGemv { k, n, m, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = module_key!(
            self,
            "batched_hw_dp4a_q6k_gemv_{}_{}_{}_{}",
            k,
            n,
            m,
            num_warps
        );

        self.ensure_kernel_module(&cache_key, &kernel_type)?;

        let module = self
            .modules
            .get_mut(&*cache_key)
            .expect("module just inserted");

        // As the single-row launch: a grid-stride loop over output rows.
        let threads = num_warps * 32;
        let grid_x = n.min(self.num_sms * 16);
        let config = LaunchConfig::grid_2d(grid_x, 1, threads, 1);

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_q8 = q8_ptr;
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All device pointers (output, weights, q8) are allocated by
        // CudaExecutor; q8 holds m Q8_1 rows of k values and output m × n f32
        // values, which is all the kernel touches. k_val and n_val are
        // stack-local scalars passed by pointer. The stream synchronizes before
        // any host-side read of the output buffer.
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

        // Record for manual graph construction, as hw_dp4a_q6k_gemv_into does:
        // a launch missing from the graph leaves its output stale on replay.
        if self.graph_recording {
            let module = self.modules.get_mut(&*cache_key).expect("module exists");
            let func = module.get_function(kernel_name)?;
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config,
                arg_data: vec![ptr_output, ptr_weights, ptr_q8, k_val as u64, n_val as u64],
            });
        }

        Ok(())
    }
}
