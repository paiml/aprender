
impl CudaExecutor {

    /// PAR-096: Batched Q4K GEMV with L2 cache reuse
    ///
    /// Performs M sequential GEMVs using the same cached weights.
    /// Weight data stays in L2 cache between rows, amortizing memory bandwidth.
    /// This enables speculative decode verification without WMMA kernel complexity.
    ///
    /// # Arguments
    /// * `weight_name` - Name of cached Q4K weight
    /// * `input` - Input activations [M, K] in FP32
    /// * `output` - Output buffer [M, N] in FP32
    /// * `m` - Batch size (number of tokens)
    /// * `k` - Input dimension (must be multiple of 256)
    /// * `n` - Output dimension
    ///
    /// # Performance
    /// Expected ~2-3x speedup over M separate calls due to L2 weight caching.
    /// Weights (3MB per layer) fit in RTX 4090 L2 (72MB).
    ///
    /// # Errors
    /// Returns error if weight not cached or kernel launch fails
    #[allow(clippy::too_many_arguments)]
    pub fn batched_q4k_gemv_cached(
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
                "PAR-096: Input size {} != expected M*K = {}*{} = {}",
                input.len(),
                m,
                k,
                expected_input
            )));
        }
        if output.len() != expected_output {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-096: Output size {} != expected M*N = {}*{} = {}",
                output.len(),
                m,
                n,
                expected_output
            )));
        }

        // Get cached weight pointer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // PAR-108: Use true batched kernel for dequant sharing across M sequences
        // This amortizes weight dequantization cost across batch, providing ~15x speedup for M=4
        let input_buf = GpuBuffer::from_host(&self.context, input)?;
        let output_buf = GpuBuffer::new(&self.context, output.len())?;

        // Single batched kernel launch - dequantizes once, multiplies by M inputs
        self.batched_q4k_gemv_into(weight_ptr, &input_buf, &output_buf, m, n, k)?;

        // Synchronize and download results
        self.stream.synchronize()?;
        output_buf.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-014: Fused FFN on GPU (up + GELU + down in single GPU round-trip)
    ///
    /// Reduces 2 GPU round-trips to 1 by keeping intermediate FFN hidden state on GPU.
    /// Input and output are CPU slices; intermediate computation stays on GPU.
    ///
    /// # Arguments
    /// * `input` - Hidden state [hidden_dim]
    /// * `output` - Output hidden state [hidden_dim]
    /// * `ffn_up_name` - Cache key for FFN up weight
    /// * `ffn_down_name` - Cache key for FFN down weight
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    #[allow(clippy::too_many_arguments)]
    pub fn fused_ffn_q4k(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        ffn_up_name: &str,
        ffn_down_name: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<(), GpuError> {
        // Verify weights are cached (ALB-098: checks pool first, then individual cache)
        let up_ptr = self.get_quantized_weight_ptr(ffn_up_name)?;
        let down_ptr = self.get_quantized_weight_ptr(ffn_down_name)?;

        // 1. Upload input to GPU (only transfer IN for FFN)
        let buf_input = GpuBuffer::from_host(&self.context, input)?;

        // 2. Allocate intermediate buffer for FFN hidden state
        let buf_intermediate = GpuBuffer::<f32>::new(&self.context, intermediate_dim as usize)?;

        // 3. Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, hidden_dim as usize)?;

        // 4. FFN up projection: [hidden_dim] -> [intermediate_dim]
        let up_kernel_type = KernelType::Q4KGemv {
            k: hidden_dim,
            n: intermediate_dim,
        };
        let up_kernel_name = self.kernels.kernel_name(&up_kernel_type);
        let up_cache_key = format!("q4k_gemv_{}_{}", hidden_dim, intermediate_dim);

        if !self.modules.contains_key(&up_cache_key) {
            let ptx = self.kernels.generate_ptx(&up_kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(up_cache_key.clone(), module);
        }

        {
            let module = self.modules.get_mut(&up_cache_key).expect("just inserted");
            let config = LaunchConfig::grid_2d(intermediate_dim, 1, 32, 1);

            let mut ptr_output = buf_intermediate.as_ptr();
            let mut ptr_weights = up_ptr;
            let mut ptr_input = buf_input.as_ptr();
            let mut k_val = hidden_dim;
            let mut n_val = intermediate_dim;

            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    module,
                    up_kernel_name,
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
        }

        // 5. GELU activation in-place on intermediate buffer
        self.gelu_gpu(&buf_intermediate, intermediate_dim)?;

        // 6. FFN down projection: [intermediate_dim] -> [hidden_dim]
        let down_kernel_type = KernelType::Q4KGemv {
            k: intermediate_dim,
            n: hidden_dim,
        };
        let down_kernel_name = self.kernels.kernel_name(&down_kernel_type);
        let down_cache_key = format!("q4k_gemv_{}_{}", intermediate_dim, hidden_dim);

        if !self.modules.contains_key(&down_cache_key) {
            let ptx = self.kernels.generate_ptx(&down_kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(down_cache_key.clone(), module);
        }

        {
            let module = self
                .modules
                .get_mut(&down_cache_key)
                .expect("just inserted");
            let config = LaunchConfig::grid_2d(hidden_dim, 1, 32, 1);

            let mut ptr_output = buf_output.as_ptr();
            let mut ptr_weights = down_ptr;
            let mut ptr_input = buf_intermediate.as_ptr();
            let mut k_val = intermediate_dim;
            let mut n_val = hidden_dim;

            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe {
                self.stream.launch_kernel(
                    module,
                    down_kernel_name,
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
        }

        // 7. Sync and download result (only transfer OUT for FFN)
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }
}

include!("silu.rs");
include!("fused_ffn.rs");
include!("rope_indirect.rs");
