
impl CudaExecutor {

    /// PAR-061: Execute Tiled Q4_K GEMV into existing buffer (zero-allocation, high-perf)
    ///
    /// Like `q4k_gemv_into` but uses TiledQ4KGemv kernel with:
    /// - 256 threads per block (vs 32 in basic kernel) for better occupancy
    /// - Shared memory caching of input vector (~8x fewer global reads)
    /// - Multiple outputs per block for better work efficiency
    ///
    /// Performance: ~5-6x faster than basic Q4KGemv on RTX 4090
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `output` - Pre-allocated output buffer (must be at least n elements)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (should be multiple of 256 for best performance)
    #[inline]
    pub fn q4k_gemv_into_tiled(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "q4k_gemv_into_tiled")?;
        // CORRECTNESS-001: Use 4 outputs per block (matches verified working q4k_gemv_cached)
        // The 8-outputs config was causing incorrect results
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)
        let outputs_per_block = 4u32;

        // PAR-502: Select kernel based on K size to avoid shared memory overflow
        let (kernel_type, cache_key) = if k > MAX_TILED_K && k.is_multiple_of(256) {
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            (kt, ck)
        } else {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            (kt, ck)
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // CORRECTNESS-001: Grid configuration matching q4k_gemv_cached
        // 128 threads per block, 4 outputs per block
        let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
        let config = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);

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

        Ok(())
    }

    /// CORRECTNESS-001: Test wrapper for q4k_gemv_into_tiled with CPU I/O
    ///
    /// Uses the exact same kernel as workspace path but with sync and CPU transfer.
    /// For debugging correctness issues.
    pub fn q4k_gemv_cached_tiled(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Get cached weight pointer
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // Upload input to GPU
        let buf_input = GpuBuffer::from_host(&self.context, input)?;

        // Create output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // Run the tiled kernel (same as workspace path)
        self.q4k_gemv_into_tiled(weight_ptr, &buf_input, &buf_output, n, k)?;

        // Sync and download
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        Ok(())
    }

    /// PAR-108: Execute Batched Q4_K GEMV for 2x Ollama target
    ///
    /// Key optimization: Sequential GEMV dequantizes weights M times for M sequences.
    /// Batched GEMV dequantizes ONCE and multiplies by M different input vectors.
    /// This amortizes ALU-bound dequantization cost (32% bandwidth → higher efficiency).
    ///
    /// Memory layout:
    /// - `input`: M × K row-major (M batch elements, K elements each)
    /// - `output`: M × N row-major (M batch elements, N outputs each)
    /// - `weights`: N × K/256 Q4_K super-blocks (shared across batch)
    ///
    /// Performance insight (Five-Whys PAR-108):
    /// 1. WHY can't batched throughput reach 2x? → 32% bandwidth efficiency
    /// 2. WHY only 32%? → Sequential GEMV dequantizes per sequence
    /// 3. WHY not share dequantization? → No batched GEMV kernel existed
    /// 4. WHY not tensor cores? → Complex WMMA PTX (~400 LOC)
    /// 5. WHY batched GEMV works? → Simpler, shares dequant in registers
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing M×K input matrix (row-major)
    /// * `output` - Pre-allocated M×N output buffer (row-major)
    /// * `m` - Batch size (number of sequences, max 8)
    /// * `n` - Output dimension (weight rows)
    /// * `k` - Input dimension (weight columns, must be multiple of 256)
    ///
    /// PAR-129 FIX: Support M>8 via tiled execution or multi-warp kernel
    /// - M=16: Uses MultiWarpBatchedQ4KGemvKernel (2 warps × 8, L1 cache sharing)
    /// - M<=8: Single kernel launch with BatchedQ4KGemvKernel
    /// - M>8 (not 16): Processes in tiles of 8
    #[inline]
    pub fn batched_q4k_gemv_into(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        validate_device_ptr(weight_ptr, "batched_q4k_gemv_into")?;
        debug_assert!(
            k.is_multiple_of(256),
            "K must be multiple of 256 for Q4K super-blocks"
        );

        // PAR-129: Use multi-warp kernel for M=16 or M=32 (optimal L1 cache sharing)
        if m == 16 {
            return self.batched_q4k_gemv_into_multi_warp(weight_ptr, input, output, n, k, 2);
        }
        if m == 32 {
            return self.batched_q4k_gemv_into_multi_warp(weight_ptr, input, output, n, k, 4);
        }

        // Tile over M for M>8 (process 8 sequences at a time)
        const MAX_TILE_M: u32 = 8;
        let num_tiles = (m + MAX_TILE_M - 1) / MAX_TILE_M;

        for tile_idx in 0..num_tiles {
            let tile_start = tile_idx * MAX_TILE_M;
            let tile_m = (m - tile_start).min(MAX_TILE_M);

            let kernel_type = KernelType::BatchedQ4KGemv { m: tile_m, k, n };
            let kernel_name = self.kernels.kernel_name(&kernel_type);
            let cache_key = format!("batched_q4k_gemv_{}_{}_{}", tile_m, k, n);

            if !self.modules.contains_key(&cache_key) {
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(cache_key.clone(), module);
            }

            let module = self
                .modules
                .get_mut(&cache_key)
                .expect("module just inserted");

            // Grid: N blocks (one per output row), 32 threads per block
            let config = LaunchConfig::grid_2d(n, 1, 32, 1);

            // Offset pointers for this tile
            // Input: tile_start * k elements into input buffer
            // Output: tile_start * n elements into output buffer
            let input_offset = (tile_start * k) as usize * std::mem::size_of::<f32>();
            let output_offset = (tile_start * n) as usize * std::mem::size_of::<f32>();

            let mut ptr_output = output.as_ptr() + output_offset as u64;
            let mut ptr_weights = weight_ptr;
            let mut ptr_input = input.as_ptr() + input_offset as u64;
            let mut k_val = k;
            let mut n_val = n;
            let mut m_val = tile_m;

            // Kernel signature: batched_q4k_gemv_warp_reduce(y_ptr, w_ptr, x_ptr, k_dim, n_dim, m_dim)
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
                        std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        Ok(())
    }

    /// PAR-129: Multi-warp batched Q4K GEMV for M=16/32
    /// Uses 2-4 warps per block, each handling 8 batch elements.
    /// All warps share L1-cached weights, avoiding weight re-reads.
    #[inline]
    fn batched_q4k_gemv_into_multi_warp(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
        warps: u32,
    ) -> Result<(), GpuError> {
        let kernel_type = KernelType::MultiWarpBatchedQ4KGemv { k, n, warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("multi_warp_batched_q4k_gemv_{}_{}_{}", k, n, warps);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: N blocks (one per output row), warps*32 threads per block
        let threads_per_block = warps * 32;
        let config = LaunchConfig::grid_2d(n, 1, threads_per_block, 1);

        // m_dim = warps * 8 (each warp handles 8 batch elements)
        let m = warps * 8;

        let mut ptr_output = output.as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut ptr_input = input.as_ptr();
        let mut k_val = k;
        let mut n_val = n;
        let mut m_val = m;

        // Kernel signature: batched_q4k_gemv_warp_reduce(y_ptr, w_ptr, x_ptr, k_dim, n_dim, m_dim)
        // Same signature as BatchedQ4KGemv since we use the same trueno kernel
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
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }
}

include!("q4k_gemv_cached_uses.rs");
include!("q6k_gemv_indexed.rs");
include!("q4k_mwv_gemv.rs");
