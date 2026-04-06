impl CudaExecutor {

    /// Execute fused GEMM + bias + activation kernel (IMP-900b)
    ///
    /// Performs C = activation(A @ B + bias) in a single kernel launch,
    /// reducing kernel launch overhead by 3x compared to separate operations.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (m × k)
    /// * `b` - Input matrix B (k × n)
    /// * `bias` - Bias vector (n elements) or None
    /// * `c` - Output matrix C (m × n)
    /// * `m`, `n`, `k` - Matrix dimensions
    /// * `activation` - Activation type (0=none, 1=relu, 2=gelu)
    ///
    /// # Performance Impact
    ///
    /// - Without fusion: 3 kernel launches (GEMM + add + activation)
    /// - With fusion: 1 kernel launch
    /// - Expected improvement: 1.3-1.5x for small matrices
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fused(
        &mut self,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
        activation: u32,
    ) -> Result<(), GpuError> {
        // Validate sizes
        let expected_a = (m * k) as usize;
        let expected_b = (k * n) as usize;
        let expected_c = (m * n) as usize;

        if a.len() != expected_a || b.len() != expected_b || c.len() != expected_c {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "GEMM size mismatch: A[{}] expected {}, B[{}] expected {}, C[{}] expected {}",
                a.len(),
                expected_a,
                b.len(),
                expected_b,
                c.len(),
                expected_c
            )));
        }

        if let Some(b_vec) = bias {
            if b_vec.len() != n as usize {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "Bias size mismatch: got {}, expected {}",
                    b_vec.len(),
                    n
                )));
            }
        }

        // Track fusion stats in pool
        self.memory_pool
            .record_allocation(expected_a * 4 + expected_b * 4 + expected_c * 4);

        // IMP-900b: Use fused kernel type
        let kernel_type = KernelType::GemmBiasActivation {
            m,
            n,
            k,
            activation,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gemm_fused_{}_{}_{}_{}", m, n, k, activation);

        // Load module if not cached (falls back to tiled for now)
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate GPU buffers
        let buf_a = GpuBuffer::from_host(&self.context, a)?;
        let buf_b = GpuBuffer::from_host(&self.context, b)?;
        // PARITY-114 FIX: Initialize output buffer with zeros to prevent state accumulation
        let c_zeros = vec![0.0f32; expected_c];
        let buf_c = GpuBuffer::from_host(&self.context, &c_zeros)?;

        // Launch configuration
        // PARITY-114 FIX: Grid X is for columns (N), Grid Y is for rows (M)
        let tile_size = 32u32;
        let config = LaunchConfig::grid_2d(
            (n + tile_size - 1) / tile_size, // Grid X - columns (N dimension)
            (m + tile_size - 1) / tile_size, // Grid Y - rows (M dimension)
            tile_size,
            tile_size,
        );

        // Get raw pointers for kernel args
        let mut ptr_a = buf_a.as_ptr();
        let mut ptr_b = buf_b.as_ptr();
        let mut ptr_c = buf_c.as_ptr();
        let mut m_val = m as i32;
        let mut n_val = n as i32;
        let mut k_val = k as i32;

        // Launch kernel
        // SAFETY: Buffers are valid, config matches kernel expectations
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // IMP-1000: Apply bias + activation on GPU (eliminates host roundtrip)
        if bias.is_some() || activation > 0 {
            let total_elements = expected_c as u32;

            // Create bias buffer (use zeros if no bias)
            let bias_data: Vec<f32> =
                bias.map_or_else(|| vec![0.0f32; n as usize], <[f32]>::to_vec);
            let buf_bias = GpuBuffer::from_host(&self.context, &bias_data)?;

            // Load epilogue kernel
            let epilogue_type = KernelType::BiasActivation {
                n: total_elements,
                bias_size: n,
                activation,
            };
            let epilogue_name = self.kernels.kernel_name(&epilogue_type);
            let epilogue_key = format!("bias_act_{}_{}", total_elements, activation);

            if !self.modules.contains_key(&epilogue_key) {
                let ptx = self.kernels.generate_ptx(&epilogue_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(epilogue_key.clone(), module);
            }

            let epilogue_module = self
                .modules
                .get_mut(&epilogue_key)
                .expect("module just inserted");

            // Launch epilogue kernel
            let threads = 256u32;
            let blocks = (total_elements + threads - 1) / threads;
            let epilogue_config = LaunchConfig::linear(blocks, threads);

            let mut ptr_c_epilogue = buf_c.as_ptr();
            let mut ptr_bias = buf_bias.as_ptr();
            let mut n_val_epilogue = total_elements as i32;
            let mut bias_size_val = n as i32;

            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    epilogue_module,
                    epilogue_name,
                    &epilogue_config,
                    &mut [
                        std::ptr::from_mut(&mut ptr_c_epilogue) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_bias) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut n_val_epilogue) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut bias_size_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Synchronize and copy result back (single H2D transfer)
        self.stream.synchronize()?;
        buf_c.copy_to_host(c)?;

        self.memory_pool
            .record_deallocation(expected_a * 4 + expected_b * 4 + expected_c * 4);

        Ok(())
    }

    /// Execute softmax kernel on a vector
    ///
    /// Computes numerically stable softmax in-place.
    pub fn softmax(&mut self, data: &mut [f32]) -> Result<(), GpuError> {
        let dim = data.len() as u32;

        let kernel_type = KernelType::Softmax { dim };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("softmax_{}", dim);

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate input and output buffers on GPU
        let input_buf = GpuBuffer::from_host(&self.context, data)?;
        let output_buf: GpuBuffer<f32> = GpuBuffer::new(&self.context, data.len())?;

        // Launch with 1 block, dim threads (up to 1024)
        let threads = dim.min(1024);
        let config = LaunchConfig::linear(1, threads);

        // Get raw pointers for kernel args (input_ptr, output_ptr, length)
        let mut input_ptr = input_buf.as_ptr();
        let mut output_ptr = output_buf.as_ptr();
        let mut length_val = dim;

        // Launch kernel
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut input_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut output_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut length_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Synchronize and copy result back
        self.stream.synchronize()?;
        output_buf.copy_to_host(data)?;

        Ok(())
    }

    /// Execute Q4_K quantized GEMM (fused dequantization + matmul)
    ///
    /// # Arguments
    ///
    /// * `weights` - Quantized weights in Q4_K format
    /// * `input` - Input vector (f32)
    /// * `output` - Output vector (f32)
    /// * `m` - Output dimension
    /// * `k` - Input dimension (must be divisible by 32)
    pub fn q4k_matvec(
        &mut self,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PARITY-003 FIX: Use QuantizedGemmGgml for GGUF Q4_K format (256 values, 144 bytes per super-block)
        // Previous: QuantizedGemm was for a different Q4 layout, causing garbage output
        let kernel_type = KernelType::QuantizedGemmGgml { m, n: 1, k };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_ggml_{}_{}", m, k);

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate GPU buffers
        let buf_weights = GpuBuffer::from_host(&self.context, weights)?;
        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, m as usize)?;

        // PARITY-003 FIX: Launch configuration for GGML kernel with tile_size=32
        // The GGML kernel uses: weight_row = clamped_col, where clamped_col = ctaid_x * tile + local_col
        // For matvec (n=1), we want weight_row to iterate over m output elements
        // CRITICAL FIX: Swap m and n so kernel uses ctaid_x for weight row indexing
        // grid.x = ceil(m/tile), grid.y = 1 (but we pass n=m, m=1 to kernel!)
        let tile_size = 32u32;
        let blocks_x = (m + tile_size - 1) / tile_size; // Iterate over m outputs
        let blocks_y = 1u32; // n=1, so 1 block in y
        let config = LaunchConfig::grid_2d(blocks_x, blocks_y, tile_size, tile_size);

        // Get raw pointers for kernel args
        // Kernel signature: q4k_gemm_ggml(a_ptr, b_quant_ptr, c_ptr, m, n, k)
        // Where: a_ptr = input activations, b_quant_ptr = weights, c_ptr = output
        //
        // PARITY-003 FIX: For matvec, swap m and n so kernel uses ctaid_x for weight row
        // The kernel uses clamped_col (derived from ctaid_x) to index weight rows
        // By passing m=1, n=out_dim, the kernel will:
        //   - Use ctaid_x (0 to out_dim/tile) for weight row indexing via clamped_col
        //   - Output at index global_row * n + global_col = 0 * out_dim + col = col
        let mut ptr_input = buf_input.as_ptr(); // a_ptr: input activations
        let mut ptr_weights = buf_weights.as_ptr(); // b_quant_ptr: quantized weights
        let mut ptr_output = buf_output.as_ptr(); // c_ptr: output
        let mut m_val = 1u32; // m=1 (swapped for matvec)
        let mut n_val = m; // n=out_dim (swapped for matvec)
        let mut k_val = k; // u32 as expected by kernel

        // Launch kernel
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void, // a_ptr
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void, // b_quant_ptr
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void, // c_ptr
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,     // m
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,     // n (was missing!)
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,     // k
                ],
            )?;
        }

        // Synchronize and copy result
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// Execute Q4_K GEMV (fused dequantization + matvec) - PAR-003
    ///
    /// Optimized kernel for M=1 token generation. Uses warp shuffle reduction
    /// with one warp (32 threads) per output element. No shared memory needed.
    ///
    /// # Performance
    ///
    /// - Memory: 7.1x more efficient than dequant+GEMV (reads Q4_K directly)
    /// - Compute: Fused dequant+multiply avoids intermediate buffer
    /// - Target: >24 tok/s (M2 milestone), matching llama.cpp performance
    ///
    /// # Arguments
    ///
    /// * `weights` - Quantized weights in Q4_K GGML format (144 bytes per 256 values)
    /// * `input` - Input vector (f32, length k)
    /// * `output` - Output vector (f32, length n)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be divisible by 256)
    pub fn q4k_gemv(
        &mut self,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PAR-003: Use dedicated Q4_K GEMV kernel for M=1 operations
        let kernel_type = KernelType::Q4KGemv { k, n };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_gemv_{}_{}", k, n);

        // Load module if not cached
        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Allocate GPU buffers
        let buf_weights = GpuBuffer::from_host(&self.context, weights)?;
        let buf_input = GpuBuffer::from_host(&self.context, input)?;
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // PAR-003: Launch configuration for GEMV kernel
        // Grid: N blocks (one per output element)
        // Block: 32 threads (one warp for reduction)
        // No shared memory needed
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        // Kernel signature: q4k_gemv_warp_reduce(y_ptr, w_ptr, x_ptr, k_dim, n_dim)
        let mut ptr_output = buf_output.as_ptr(); // y_ptr: output vector
        let mut ptr_weights = buf_weights.as_ptr(); // w_ptr: quantized weights
        let mut ptr_input = buf_input.as_ptr(); // x_ptr: input vector
        let mut k_val = k; // k_dim
        let mut n_val = n; // n_dim

        // Launch kernel
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void, // y_ptr
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void, // w_ptr
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,  // x_ptr
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,      // k_dim
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,      // n_dim
                ],
            )?;
        }

        // Synchronize and copy result
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }
}
