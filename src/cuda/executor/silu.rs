impl CudaExecutor {
    // =========================================================================
    // PAR-023: Activation and Element-wise GPU Operations
    // =========================================================================

    /// PAR-023: SiLU activation on GPU buffer
    ///
    /// Computes: output[i] = input[i] * sigmoid(input[i])
    ///
    /// # Returns
    ///
    /// GPU buffer with activated result (no sync - async)
    pub fn silu_gpu(&mut self, input: &GpuBuffer<f32>, n: u32) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::Silu { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("silu_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // 256 threads per block for element-wise ops
        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }

    /// PAR-023: GELU activation on GPU buffer (async, returns new buffer)
    ///
    /// Computes approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    ///
    /// Unlike `gelu_gpu`, this returns a new buffer for async pipeline use.
    ///
    /// # Returns
    ///
    /// GPU buffer with activated result (no sync - async)
    pub fn gelu_async(
        &mut self,
        input: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::Gelu { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gelu_async_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }

    /// PAR-023: Element-wise multiply on GPU buffers
    ///
    /// Computes: output[i] = input1[i] * input2[i]
    /// Used for gated activations in SwiGLU.
    ///
    /// # Returns
    ///
    /// GPU buffer with product (no sync - async)
    pub fn elementwise_mul_gpu(
        &mut self,
        input1: &GpuBuffer<f32>,
        input2: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::ElementwiseMul { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("elementwise_mul_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

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

        Ok(output)
    }

    /// PAR-023: Fused SwiGLU activation on GPU buffers
    ///
    /// Computes: output[i] = silu(gate[i]) * up[i]
    /// Combines SiLU activation and multiply in one memory pass.
    ///
    /// # Returns
    ///
    /// GPU buffer with activated result (no sync - async)
    pub fn fused_swiglu_gpu(
        &mut self,
        gate: &GpuBuffer<f32>,
        up: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::FusedSwiglu { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_swiglu_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_gate = gate.as_ptr();
        let mut ptr_up = up.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_gate) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }

    /// PAR-044: Fused SwiGLU into existing buffer (zero-allocation, async)
    #[inline]
    pub fn fused_swiglu_into(
        &mut self,
        gate: &GpuBuffer<f32>,
        up: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedSwiglu { n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_swiglu_{}", n);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_gate = gate.as_ptr();
        let mut ptr_up = up.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut n_val = n;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_gate) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_up) as *mut std::ffi::c_void,
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
                arg_data: vec![ptr_gate, ptr_up, ptr_output, n_val as u64],
            });
        }

        Ok(())
    }

    /// PMAT-PERF-009: Fused Q/K/V projection on GPU
    ///
    /// Computes Q, K, V projections in a single kernel launch.
    /// Reduces kernel launch overhead from 3 launches to 1.
    ///
    /// # Arguments
    ///
    /// * `x` - Input hidden state [hidden_size]
    /// * `w_q` - Query weight matrix [hidden_size, hidden_size]
    /// * `w_k` - Key weight matrix [hidden_size, kv_dim]
    /// * `w_v` - Value weight matrix [hidden_size, kv_dim]
    /// * `out_q` - Output Q buffer [hidden_size]
    /// * `out_k` - Output K buffer [kv_dim]
    /// * `out_v` - Output V buffer [kv_dim]
    /// * `hidden_size` - Hidden dimension
    /// * `kv_dim` - KV dimension (for GQA, may differ from hidden_size)
    pub fn fused_qkv_into(
        &mut self,
        x: &GpuBuffer<f32>,
        w_q: &GpuBuffer<f32>,
        w_k: &GpuBuffer<f32>,
        w_v: &GpuBuffer<f32>,
        out_q: &GpuBuffer<f32>,
        out_k: &GpuBuffer<f32>,
        out_v: &GpuBuffer<f32>,
        hidden_size: u32,
        kv_dim: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedQKV {
            hidden_size,
            kv_dim,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_qkv_{}_{}", hidden_size, kv_dim);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: one block per output row (max of hidden_size for Q, kv_dim for K/V)
        // Block: 32 threads (one warp) per row
        let rows = hidden_size.max(kv_dim);
        let config = LaunchConfig::grid_2d(rows, 1, 32, 1);

        let mut ptr_x = x.as_ptr();
        let mut ptr_wq = w_q.as_ptr();
        let mut ptr_wk = w_k.as_ptr();
        let mut ptr_wv = w_v.as_ptr();
        let mut ptr_out_q = out_q.as_ptr();
        let mut ptr_out_k = out_k.as_ptr();
        let mut ptr_out_v = out_v.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_x) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_wq) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_wk) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_wv) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_out_q) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_out_k) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_out_v) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }
}
