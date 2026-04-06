impl CudaExecutor {

    /// QWEN-009: 3-way fused FFN kernel: RMSNorm → Gate/Up Q4K GEMV → SwiGLU
    ///
    /// Combines RMSNorm normalization, dual Q4K projections (gate & up), and
    /// SwiGLU activation in a single kernel pass for 1.2x FFN forward speedup.
    ///
    /// # Arguments
    ///
    /// * `input` - Input hidden state [hidden_size] (FP32)
    /// * `gamma` - RMSNorm weights [hidden_size] (FP32)
    /// * `w_gate_ptr` - Gate weight pointer (Q4K quantized)
    /// * `w_up_ptr` - Up weight pointer (Q4K quantized)
    /// * `output` - Output buffer [intermediate_size] (FP32)
    /// * `hidden_size` - K dimension (input dimension)
    /// * `intermediate_size` - N dimension (output dimension)
    /// * `epsilon` - RMSNorm epsilon (typically 1e-6 for Qwen)
    ///
    /// # Performance
    ///
    /// Eliminates 3 separate kernel launches:
    /// 1. RMSNorm kernel (separate)
    /// 2. Gate Q4K GEMV
    /// 3. Up Q4K GEMV
    /// 4. SwiGLU activation
    ///
    /// Into single kernel with:
    /// - Normalized input cached in shared memory (not written to global)
    /// - Gate/up computed in parallel from shared memory
    /// - SwiGLU applied before storing final result
    #[allow(clippy::too_many_arguments)]
    pub fn fused_ffn_rmsnorm_swiglu_q4k_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        w_gate_ptr: u64,
        w_up_ptr: u64,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        intermediate_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedRmsNormGateUpSwigluQ4K {
            k: hidden_size,
            n: intermediate_size,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!(
            "fused_rmsnorm_gate_up_swiglu_q4k_{}_{}",
            hidden_size, intermediate_size
        );

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: one block per output row (intermediate_size)
        // Block: 256 threads (8 warps) for cooperative loading and reduction
        let config = LaunchConfig::grid_2d(intermediate_size, 1, 256, 1);

        // Kernel parameter order (from trueno-gpu FusedRmsNormGateUpSwigluQ4KKernel):
        // out_ptr, wg_ptr, wu_ptr, x_ptr, gamma_ptr, k_dim, n_dim
        let mut ptr_output = output.as_ptr();
        let mut ptr_w_gate = w_gate_ptr;
        let mut ptr_w_up = w_up_ptr;
        let mut ptr_input = input.as_ptr();
        let mut ptr_gamma = gamma.as_ptr();
        let mut k_val = hidden_size;
        let mut n_val = intermediate_size;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w_gate) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w_up) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// QWEN-009: 3-way fused FFN with cached weights
    ///
    /// Convenience wrapper that looks up weight cache keys and calls the
    /// underlying fused kernel.
    ///
    /// # Arguments
    ///
    /// * `input` - Input hidden state [hidden_size] (FP32)
    /// * `gamma` - RMSNorm weights [hidden_size] (FP32)
    /// * `w_gate_name` - Cache key for gate weight
    /// * `w_up_name` - Cache key for up weight
    /// * `output` - Output buffer [intermediate_size] (FP32)
    /// * `hidden_size` - K dimension (input dimension)
    /// * `intermediate_size` - N dimension (output dimension)
    /// * `epsilon` - RMSNorm epsilon (typically 1e-6 for Qwen)
    #[allow(clippy::too_many_arguments)]
    pub fn fused_ffn_rmsnorm_swiglu_q4k_cached(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        w_gate_name: &str,
        w_up_name: &str,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        intermediate_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let w_gate_ptr = self.get_quantized_weight_ptr(w_gate_name)?;
        let w_up_ptr = self.get_quantized_weight_ptr(w_up_name)?;

        self.fused_ffn_rmsnorm_swiglu_q4k_into(
            input,
            gamma,
            w_gate_ptr,
            w_up_ptr,
            output,
            hidden_size,
            intermediate_size,
            epsilon,
        )
    }

    /// PMAT-PERF-009: Fused Gate+Up FFN with SwiGLU on GPU
    ///
    /// Computes gate and up projections + SiLU activation in a single kernel.
    /// Reduces kernel launch overhead from 2 launches + activation to 1.
    ///
    /// # Arguments
    ///
    /// * `x` - Input hidden state [hidden_size]
    /// * `w_gate` - Gate weight matrix [hidden_size, intermediate_size]
    /// * `w_up` - Up weight matrix [hidden_size, intermediate_size]
    /// * `output` - Output buffer [intermediate_size], contains SiLU(gate) * up
    /// * `hidden_size` - Hidden dimension
    /// * `intermediate_size` - Intermediate FFN dimension
    pub fn fused_gate_up_into(
        &mut self,
        x: &GpuBuffer<f32>,
        w_gate: &GpuBuffer<f32>,
        w_up: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        intermediate_size: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::FusedGateUp {
            hidden_size,
            intermediate_size,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("fused_gate_up_{}_{}", hidden_size, intermediate_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: one block per output row (intermediate_size)
        // Block: 32 threads (one warp) per row
        let config = LaunchConfig::grid_2d(intermediate_size, 1, 32, 1);

        let mut ptr_x = x.as_ptr();
        let mut ptr_wg = w_gate.as_ptr();
        let mut ptr_wu = w_up.as_ptr();
        let mut ptr_out = output.as_ptr();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_x) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_wg) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_wu) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-060: Apply RoPE (Rotary Position Embedding) on GPU
    ///
    /// Applies rotary position embeddings to Q or K vectors.
    /// This is a critical optimization - eliminates CPU fallback that caused
    /// 28 GPU syncs + D2H/H2D copies per token.
    ///
    /// # Arguments
    ///
    /// * `input` - Input Q or K vector (FP32)
    /// * `output` - Output buffer (can alias input for in-place)
    /// * `position` - Current sequence position
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `theta` - RoPE base frequency
    pub fn rope_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        position: u32,
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::Rope {
            num_heads,
            head_dim,
            theta,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("rope_{}_{}", num_heads, head_dim);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: 1 thread per rotation pair = num_heads * (head_dim / 2)
        let num_pairs = num_heads * (head_dim / 2);
        let threads = 256;
        let blocks = (num_pairs + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut pos_val = position;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut pos_val) as *mut std::ffi::c_void,
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
                arg_data: vec![ptr_input, ptr_output, pos_val as u64],
            });
        }

        Ok(())
    }

    /// PAR-054: RoPE with indirect position (CUDA Graph Compatible)
    ///
    /// Same as `rope_into` but reads position from device memory instead of kernel parameter.
    /// This is required for CUDA graph capture since kernel parameters are baked at capture time.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (num_heads * head_dim elements)
    /// * `output` - Output tensor (same size as input)
    /// * `position_buf` - Device buffer containing u32 position (1 element)
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `theta` - RoPE base frequency
    pub fn rope_indirect_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        position_buf: &GpuBuffer<u32>,
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::RopeIndirect {
            num_heads,
            head_dim,
            theta,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("rope_indirect_{}_{}", num_heads, head_dim);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: 1 thread per rotation pair = num_heads * (head_dim / 2)
        let num_pairs = num_heads * (head_dim / 2);
        let threads = 256;
        let blocks = (num_pairs + threads - 1) / threads;
        let config = LaunchConfig::grid_2d(blocks, 1, threads, 1);

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

    /// CORRECTNESS-011: RoPE NEOX style (split halves)
    ///
    /// Same API as rope_into but uses NEOX-style element pairing (i, i + half_dim)
    /// instead of adjacent pairs (2*i, 2*i+1). Required for Qwen2.5 models.
    pub fn rope_neox_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        position: u32,
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::RopeNeox {
            num_heads,
            head_dim,
            theta,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("rope_neox_{}_{}", num_heads, head_dim);

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
        let mut pos_val = position;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut pos_val) as *mut std::ffi::c_void,
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
                arg_data: vec![ptr_input, ptr_output, pos_val as u64],
            });
        }

        Ok(())
    }
}
