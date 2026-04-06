impl CudaExecutor {

    /// PAR-014: Apply LayerNorm on GPU
    ///
    /// Performs: output = (input - mean) / sqrt(var + eps) * gamma + beta
    /// Part of persistent GPU tensor optimization for M4 milestone.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_gpu(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        beta: &GpuBuffer<f32>,
        hidden_size: u32,
        batch_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::LayerNorm {
            hidden_size,
            epsilon,
            affine: true,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("layernorm_{}_{}", hidden_size, batch_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // LayerNorm uses one warp per row
        let config = LaunchConfig::grid_2d(batch_size, 1, 32, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_gamma = gamma.as_ptr();
        let mut ptr_beta = beta.as_ptr();
        let mut hidden_size_val = hidden_size;
        let mut batch_size_val = batch_size;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_beta) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut hidden_size_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut batch_size_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // No sync - caller can batch operations
        Ok(())
    }

    /// PAR-023: RMSNorm on GPU (async, no sync)
    ///
    /// RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * gamma
    ///
    /// # Arguments
    ///
    /// * `input` - GPU buffer with input vector [hidden_size]
    /// * `gamma` - GPU buffer with scale weights [hidden_size]
    /// * `hidden_size` - Dimension of the vector
    /// * `epsilon` - Numerical stability constant (default: 1e-5)
    ///
    /// # Returns
    ///
    /// GPU buffer with normalized output (no sync - async)
    pub fn rmsnorm_gpu(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        hidden_size: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let kernel_type = KernelType::RmsNorm {
            hidden_size,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("rmsnorm_{}", hidden_size);

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

        // RMSNorm uses one warp (32 threads)
        let config = LaunchConfig::grid_2d(1, 1, 32, 1);

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
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-023: NO sync - async operation for pipeline
        Ok(output)
    }

    /// PAR-044: RMSNorm into existing buffer (zero-allocation, async)
    ///
    /// Like `rmsnorm_gpu` but writes into a pre-allocated output buffer.
    ///
    /// PAR-081: Uses VectorizedRmsNorm with 256 threads for ~8x speedup
    /// over single-warp kernel (23µs → ~3µs for hidden_size=1536)
    ///
    /// CORRECTNESS-013: When CORRECTNESS_MODE=1, uses PreciseRmsNorm kernel
    /// with Kahan summation and Newton-Raphson rsqrt for CPU-matching precision.
    #[inline]
    pub fn rmsnorm_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // GH-559 DIAGNOSTIC: CPU RMSNorm bypass for Blackwell
        static CPU_RMSNORM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_cpu_rmsnorm = *CPU_RMSNORM.get_or_init(|| {
            let mode = std::env::var("CPU_RMSNORM")
                .map(|v| v == "1")
                .unwrap_or(false);
            if mode {
                eprintln!("[GH-559] CPU_RMSNORM=1: RMSNorm computed on CPU (diagnostic bypass)");
            }
            mode
        });

        if use_cpu_rmsnorm {
            self.stream.synchronize()?;
            let n = hidden_size as usize;
            let mut input_host = vec![0.0f32; n];
            let mut gamma_host = vec![0.0f32; n];
            input.copy_to_host(&mut input_host)?;
            gamma.copy_to_host(&mut gamma_host)?;
            let sq_sum: f32 = input_host.iter().map(|x| x * x).sum();
            let rms = (sq_sum / n as f32 + epsilon).sqrt();
            let mut output_host = vec![0.0f32; n];
            for i in 0..n {
                output_host[i] = (input_host[i] / rms) * gamma_host[i];
            }
            // Copy result to GPU output buffer
            let temp = GpuBuffer::from_host(&self.context, &output_host)?;
            let zeros = GpuBuffer::<f32>::new(&self.context, n)?;
            self.residual_add_into(&temp, &zeros, output, hidden_size)?;
            self.stream.synchronize()?;
            return Ok(());
        }

        // CORRECTNESS-013: Check if precise mode is requested
        static PRECISE_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_precise = *PRECISE_MODE.get_or_init(|| {
            let mode = std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false);
            if mode {
                eprintln!(
                    "[CORRECTNESS-013] RMSNorm using PreciseRmsNormKernel (Kahan+Newton-Raphson)"
                );
            }
            mode
        });

        // GH-559: On Blackwell sm_121, use single-warp RmsNorm (32 threads)
        // to isolate whether the vectorized (256 thread) kernel has a bug.
        let use_simple = self.gpu_profile.cc >= 120;

        // Choose kernel type based on mode
        let (kernel_type, cache_key) = if use_simple {
            (
                KernelType::RmsNorm {
                    hidden_size,
                    epsilon,
                },
                format!("rmsnorm_simple_{}", hidden_size),
            )
        } else if use_precise {
            (
                KernelType::PreciseRmsNorm {
                    hidden_size,
                    epsilon,
                },
                format!("rmsnorm_precise_{}", hidden_size),
            )
        } else {
            (
                KernelType::VectorizedRmsNorm {
                    hidden_size,
                    epsilon,
                },
                format!("rmsnorm_vectorized_{}", hidden_size),
            )
        };

        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            if use_simple {
                eprintln!("[GH-559] RmsNorm PTX ({} bytes)", ptx.len());
            }
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-081: 256 threads for vectorized, 32 threads for simple
        let threads = if use_simple { 32 } else { 256 };
        let config = LaunchConfig::grid_2d(1, 1, threads, 1);

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
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
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
                arg_data: vec![ptr_input, ptr_output, ptr_gamma],
            });
        }

        Ok(())
    }

    /// GH-280: Per-head QK RMSNorm for Qwen3 (one warp per head).
    ///
    /// Applies RMSNorm independently to each attention head. Gamma weights
    /// are `[head_dim]` and shared across all heads (no head offset).
    ///
    /// Grid: (num_heads, 1, 1), Block: (32, 1, 1).
    ///
    /// # Arguments
    ///
    /// * `input` - GPU buffer with Q or K: `[num_heads * head_dim]`
    /// * `gamma` - GPU buffer with norm weights: `[head_dim]`
    /// * `output` - GPU buffer for result: `[num_heads * head_dim]`
    /// * `head_dim` - Elements per head (128 for Qwen3)
    /// * `num_heads` - Number of heads (32 for Q, 8 for K)
    /// * `epsilon` - Numerical stability constant (1e-6 for Qwen3)
    pub fn per_head_rmsnorm_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        head_dim: u32,
        num_heads: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::PerHeadRmsNorm {
            head_dim,
            num_heads,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("per_head_rmsnorm_{}_{}", head_dim, num_heads);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // One warp (32 threads) per head, one block per head
        let config = LaunchConfig::grid_2d(num_heads, 1, 32, 1);

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
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-112: Batched RMSNorm for M sequences in parallel
    ///
    /// Processes M sequences in a single kernel launch using Grid.y = M.
    /// Achieves ~4x speedup over M sequential kernel launches by eliminating
    /// kernel launch overhead.
    ///
    /// # Arguments
    ///
    /// * `input` - GPU buffer with packed input [M × hidden_size]
    /// * `gamma` - GPU buffer with gamma weights [hidden_size] (shared across sequences)
    /// * `output` - GPU buffer for packed output [M × hidden_size]
    /// * `hidden_size` - Hidden dimension size
    /// * `batch_size` - Number of sequences (M)
    /// * `epsilon` - Numerical stability constant (default: 1e-5)
    pub fn batched_rmsnorm_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        batch_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::BatchedVectorizedRmsNorm {
            hidden_size,
            batch_size,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        // GH-129: PTX depends on hidden_size + epsilon (immediates) but NOT batch_size (grid dim only).
        // Remove batch_size from cache key to prevent JIT recompilation per prompt length.
        let cache_key = format!("batched_rmsnorm_vectorized_{}", hidden_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-112: Grid (1, M, 1) with 256 threads per block
        let config = LaunchConfig::grid_2d(1, batch_size, 256, 1);

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
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-112: Batched RMSNorm using raw pointer for gamma (compatible with indexed weights)
    ///
    /// Same as `batched_rmsnorm_into` but accepts gamma as raw device pointer.
    pub fn batched_rmsnorm_ptr_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma_ptr: u64,
        gamma_len: usize,
        output: &GpuBuffer<f32>,
        hidden_size: u32,
        batch_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let gamma = unsafe { GpuBuffer::from_raw_parts(gamma_ptr, gamma_len) };
        self.batched_rmsnorm_into(input, &gamma, output, hidden_size, batch_size, epsilon)?;
        std::mem::forget(gamma);
        Ok(())
    }

    /// PAR-114: Batched RoPE kernel for M sequences
    ///
    /// Applies rotary position embeddings to M sequences in parallel.
    /// Reduces 2M kernel launches to 2 (one for Q, one for K).
    ///
    /// # Arguments
    ///
    /// * `input` - Packed Q or K vectors [M × num_heads × head_dim]
    /// * `output` - Output vectors (can alias input for in-place)
    /// * `positions_buf` - GPU buffer of M positions
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `batch_size` - Number of sequences (M)
    /// * `theta` - RoPE theta base (typically 10000.0)
    #[allow(clippy::too_many_arguments)]
    pub fn batched_rope_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        positions_buf: &GpuBuffer<u32>,
        num_heads: u32,
        head_dim: u32,
        batch_size: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::BatchedRope {
            num_heads,
            head_dim,
            batch_size,
            theta,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        // GH-129: PTX depends on num_heads, head_dim, theta (immediates) but NOT batch_size (grid dim).
        // Remove batch_size from cache key to prevent JIT recompilation per prompt length.
        let cache_key = format!("batched_rope_{}_{}", num_heads, head_dim);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // PAR-114: Grid (num_heads, batch_size, 1) with head_dim/2 threads
        let threads = (head_dim / 2).min(256);
        let config = LaunchConfig::grid_2d(num_heads, batch_size, threads, 1);

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_positions = positions_buf.as_ptr();

        // SAFETY: Pointers derived from valid GpuBuffer refs, kernel config matches data dimensions
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_positions) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PAR-114: Batched Residual Add kernel for M sequences
    ///
    /// Element-wise addition for M sequences in parallel.
    /// Reduces 2M kernel launches to 2 (attention residual, FFN residual).
    ///
    /// # Arguments
    ///
    /// * `input1` - First packed input [M × n]
    /// * `input2` - Second packed input [M × n]
    /// * `output` - Output [M × n]
    /// * `n` - Elements per sequence
    /// * `batch_size` - Number of sequences (M)
    pub fn batched_residual_add_into(
        &mut self,
        input1: &GpuBuffer<f32>,
        input2: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        batch_size: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::BatchedResidualAdd { n, batch_size };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        // GH-129: PTX depends on n (immediate constant) but NOT batch_size (grid dim only).
        // Remove batch_size from cache key to prevent JIT recompilation per prompt length.
        let cache_key = format!("batched_residual_add_{}", n);

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

        let mut ptr_input1 = input1.as_ptr();
        let mut ptr_input2 = input2.as_ptr();
        let mut ptr_output = output.as_ptr();

        // SAFETY: Pointers derived from valid GpuBuffer refs, kernel config matches data dimensions
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input1) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input2) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-092: Batched fused residual add + RMSNorm
    ///
    /// Fuses `batched_residual_add_into` + `batched_rmsnorm_ptr_into` into a single kernel.
    /// Saves 28 kernel launches per decode step (1 per layer × 28 layers).
    ///
    /// residual_out[m] = residual[m] + input[m]  (for residual stream)
    /// normed_out[m]   = rmsnorm(residual_out[m], gamma, epsilon)  (for FFN projections)
    ///
    /// Grid: (1, batch_size, 1), Block: (256, 1, 1)
    #[allow(clippy::too_many_arguments)]
    pub fn batched_fused_residual_rmsnorm_into(
        &mut self,
        residual: &GpuBuffer<f32>,
        input: &GpuBuffer<f32>,
        residual_out: &GpuBuffer<f32>,
        normed_out: &GpuBuffer<f32>,
        gamma_ptr: u64,
        gamma_len: usize,
        hidden_size: u32,
        batch_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::BatchedFusedResidualRmsNorm {
            hidden_size,
            batch_size,
            epsilon,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        // PTX depends on hidden_size + epsilon (immediates) but NOT batch_size (grid dim only).
        let cache_key = format!("batched_fused_residual_rmsnorm_{}", hidden_size);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid (1, M, 1) with 256 threads per block
        let config = LaunchConfig::grid_2d(1, batch_size, 256, 1);

        // SAFETY: gamma_ptr is valid GPU allocation from indexed layer weights
        let gamma = unsafe { GpuBuffer::<f32>::from_raw_parts(gamma_ptr, gamma_len) };

        let mut ptr_residual = residual.as_ptr();
        let mut ptr_input = input.as_ptr();
        let mut ptr_res_out = residual_out.as_ptr();
        let mut ptr_norm_out = normed_out.as_ptr();
        let mut ptr_gamma = gamma.as_ptr();

        // SAFETY: All pointers derived from valid GpuBuffer refs
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_residual) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_res_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_norm_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_gamma) as *mut std::ffi::c_void,
                ],
            )?;
        }

        std::mem::forget(gamma);
        Ok(())
    }

    /// PMAT-046: Batched bias broadcast add — adds bias[dim] to packed[M×dim] in-place.
    ///
    /// Replaces M sequential residual_add_into calls with a single kernel launch.
    /// Grid: (ceil(dim/256), M, 1), Block: (256, 1, 1)
    /// Each thread: packed[blockIdx.y * dim + blockIdx.x * 256 + threadIdx.x] += bias[blockIdx.x * 256 + threadIdx.x]
    pub fn batched_bias_broadcast_add(
        &mut self,
        packed_ptr: u64,
        bias_ptr: u64,
        dim: u32,
        m: u32,
    ) -> Result<(), GpuError> {
        let cache_key = format!("batched_bias_bcast_{}", dim);

        if !self.modules.contains_key(&cache_key) {
            let ptx = format!(
                r#".version 7.0
.target sm_70
.address_size 64

.visible .entry batched_bias_broadcast_add_{dim}(
    .param .u64 packed,
    .param .u64 bias
) {{
    .reg .u64 %rd<8>;
    .reg .u32 %r<6>;
    .reg .f32 %f<3>;
    .reg .pred %p;

    // idx = blockIdx.x * 256 + threadIdx.x
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r0, 8;       // blockIdx.x * 256
    add.u32 %r2, %r2, %r1;     // idx = blockIdx.x * 256 + threadIdx.x

    // bounds check: idx < dim
    setp.ge.u32 %p, %r2, {dim};
    @%p bra DONE;

    // seq_idx = blockIdx.y
    mov.u32 %r3, %ctaid.y;

    // packed_offset = (seq_idx * dim + idx) * 4
    mul.lo.u32 %r4, %r3, {dim};
    add.u32 %r4, %r4, %r2;
    mul.wide.u32 %rd0, %r4, 4;

    // bias_offset = idx * 4
    mul.wide.u32 %rd1, %r2, 4;

    // Load pointers
    ld.param.u64 %rd2, [packed];
    ld.param.u64 %rd3, [bias];
    add.u64 %rd4, %rd2, %rd0;
    add.u64 %rd5, %rd3, %rd1;

    // packed[offset] += bias[idx]
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    add.f32 %f2, %f0, %f1;
    st.global.f32 [%rd4], %f2;

DONE:
    ret;
}}"#,
                dim = dim
            );
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let blocks_x = (dim + 255) / 256;
        let config = LaunchConfig {
            grid: (blocks_x, m, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        };

        let mut p_packed = packed_ptr;
        let mut p_bias = bias_ptr;
        let kernel_name = format!("batched_bias_broadcast_add_{}", dim);

        // SAFETY: packed_ptr is valid M×dim GPU alloc, bias_ptr is valid dim GPU alloc
        unsafe {
            self.stream.launch_kernel(
                module,
                &kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut p_packed) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut p_bias) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-046: Batched NEOX RoPE for M sequences in a single kernel launch.
    ///
    /// Replaces M×2 sequential rope_neox_into calls (Q and K separately) with
    /// a single launch per buffer. Grid: (num_heads, M, 1), Block: (half_dim, 1, 1).
    ///
    /// NEOX RoPE rotates split halves: (x[i], x[i + dim/2]) unlike standard
    /// adjacent-pair RoPE.
    pub fn batched_rope_neox_into(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        positions_buf: &GpuBuffer<u32>,
        num_heads: u32,
        head_dim: u32,
        batch_size: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        let half_dim = head_dim / 2;
        let cache_key = format!("batched_rope_neox_{}_{}", num_heads, head_dim);

        if !self.modules.contains_key(&cache_key) {
            // Precompute log2(theta) for ex2-based frequency calculation
            let log2_theta = theta.log2();
            let stride = num_heads * head_dim; // elements per sequence

            let ptx = format!(
                r#".version 7.0
.target sm_70
.address_size 64

.visible .entry batched_rope_neox_{num_heads}_{head_dim}(
    .param .u64 input,
    .param .u64 output,
    .param .u64 positions
) {{
    .reg .u64 %rd<12>;
    .reg .u32 %r<10>;
    .reg .f32 %f<16>;
    .reg .pred %p;

    // tid = threadIdx.x (0..half_dim-1)
    mov.u32 %r0, %tid.x;
    // head_idx = blockIdx.x
    mov.u32 %r1, %ctaid.x;
    // seq_idx = blockIdx.y
    mov.u32 %r2, %ctaid.y;

    // bounds check
    setp.ge.u32 %p, %r0, {half_dim};
    @%p bra DONE;

    // Load position from positions[seq_idx]
    ld.param.u64 %rd0, [positions];
    mul.wide.u32 %rd1, %r2, 4;        // seq_idx * 4
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r3, [%rd2];        // position

    // Compute frequency: freq = position * theta^(-2*tid/dim)
    // = position * 2^(-2*tid/dim * log2(theta))
    cvt.rn.f32.u32 %f0, %r0;         // tid as float
    mul.f32 %f0, %f0, 0f{neg2_over_dim:08X};  // tid * (-2/dim)
    mul.f32 %f0, %f0, 0f{log2_theta:08X};     // * log2(theta)
    ex2.approx.f32 %f1, %f0;                  // theta^(-2*tid/dim)
    cvt.rn.f32.u32 %f2, %r3;                  // position as float
    mul.f32 %f3, %f2, %f1;                    // freq = position * inv_freq

    // cos/sin via ex2 + polynomial
    cos.approx.f32 %f4, %f3;          // cos(freq)
    sin.approx.f32 %f5, %f3;          // sin(freq)

    // Compute base offset: seq_idx * stride + head_idx * head_dim
    mul.lo.u32 %r4, %r2, {stride};    // seq_idx * stride
    mul.lo.u32 %r5, %r1, {head_dim};  // head_idx * head_dim
    add.u32 %r4, %r4, %r5;            // base = seq*stride + head*head_dim

    // lo_idx = base + tid, hi_idx = base + tid + half_dim
    add.u32 %r6, %r4, %r0;            // lo_idx = base + tid
    add.u32 %r7, %r6, {half_dim};     // hi_idx = base + tid + half_dim

    // Load input[lo_idx] and input[hi_idx]
    ld.param.u64 %rd3, [input];
    mul.wide.u32 %rd4, %r6, 4;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.f32 %f6, [%rd5];        // x_lo = input[lo_idx]

    mul.wide.u32 %rd6, %r7, 4;
    add.u64 %rd7, %rd3, %rd6;
    ld.global.f32 %f7, [%rd7];        // x_hi = input[hi_idx]

    // Rotate: out_lo = x_lo * cos - x_hi * sin
    //         out_hi = x_hi * cos + x_lo * sin
    mul.f32 %f8, %f6, %f4;            // x_lo * cos
    mul.f32 %f9, %f7, %f5;            // x_hi * sin
    sub.f32 %f10, %f8, %f9;           // out_lo

    mul.f32 %f11, %f7, %f4;           // x_hi * cos
    mul.f32 %f12, %f6, %f5;           // x_lo * sin
    add.f32 %f13, %f11, %f12;         // out_hi

    // Store output
    ld.param.u64 %rd8, [output];
    mul.wide.u32 %rd9, %r6, 4;
    add.u64 %rd10, %rd8, %rd9;
    st.global.f32 [%rd10], %f10;

    mul.wide.u32 %rd9, %r7, 4;
    add.u64 %rd11, %rd8, %rd9;
    st.global.f32 [%rd11], %f13;

DONE:
    ret;
}}"#,
                num_heads = num_heads,
                head_dim = head_dim,
                half_dim = half_dim,
                stride = stride,
                neg2_over_dim = (-2.0f32 / head_dim as f32).to_bits(),
                log2_theta = log2_theta.to_bits(),
            );
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let config = LaunchConfig {
            grid: (num_heads, batch_size, 1),
            block: (half_dim, 1, 1),
            shared_mem: 0,
        };

        let mut ptr_input = input.as_ptr();
        let mut ptr_output = output.as_ptr();
        let mut ptr_positions = positions_buf.as_ptr();
        let kernel_name = format!("batched_rope_neox_{}_{}", num_heads, head_dim);

        // SAFETY: input/output are M×num_heads×head_dim GPU allocs, positions is M GPU alloc
        unsafe {
            self.stream.launch_kernel(
                module,
                &kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_positions) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }
}
