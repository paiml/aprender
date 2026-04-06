
impl CudaExecutor {

    /// FP16 Tensor Core GEMM using WMMA intrinsics (IMP-1000a)
    ///
    /// Computes C = A × B using FP16 tensor cores with FP32 accumulation.
    /// RTX 4090: 330 TFLOPS FP16 vs 83 TFLOPS FP32 (4x theoretical speedup).
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A as FP32 (will be converted to FP16)
    /// * `b` - Weight matrix B as FP32 (will be converted to FP16)
    /// * `c` - Output matrix C (FP32 accumulator)
    /// * `m`, `n`, `k` - Matrix dimensions (must be multiples of 16)
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are not multiples of 16 or kernel fails.
    pub fn gemm_fp16(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Validate dimensions are multiples of 16 (WMMA requirement)
        if !m.is_multiple_of(16) || !n.is_multiple_of(16) || !k.is_multiple_of(16) {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "FP16 Tensor Core requires dimensions multiple of 16: m={}, n={}, k={}",
                m, n, k
            )));
        }

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

        // Track memory usage
        self.memory_pool
            .record_allocation(expected_a * 4 + expected_b * 4 + expected_c * 4);

        // For now, use tiled GEMM as placeholder (FP16 WMMA PTX is generated but
        // actual tensor core execution requires half-precision buffer support)
        // The API is ready for when trueno-gpu adds FP16 buffer support
        let kernel_type = KernelType::GemmTiled {
            m,
            n,
            k,
            tile_size: 32,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gemm_fp16_{}_{}_{}", m, n, k);

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
        let buf_a = GpuBuffer::from_host(&self.context, a)?;
        let buf_b = GpuBuffer::from_host(&self.context, b)?;
        // PARITY-114 FIX: Initialize output buffer with zeros to prevent state accumulation
        let c_zeros = vec![0.0f32; expected_c];
        let buf_c = GpuBuffer::from_host(&self.context, &c_zeros)?;

        // Launch configuration (16x16 tiles for FP16)
        // PARITY-114 FIX: Grid X is for columns (N), Grid Y is for rows (M)
        let config = LaunchConfig::grid_2d((n + 31) / 32, (m + 31) / 32, 32, 32);

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

        // Synchronize and copy result back
        self.stream.synchronize()?;
        buf_c.copy_to_host(c)?;

        Ok(())
    }

    /// Compute attention score statistics (for debugging/profiling)
    #[must_use]
    pub fn flash_attention_memory_bytes(seq_len: u32, _head_dim: u32) -> (u64, u64) {
        // Naive: full N×N attention matrix
        let naive = u64::from(seq_len) * u64::from(seq_len) * 4;

        // FlashAttention: only block-sized working memory
        // Block size 64 is typical
        let block_size = 64u64;
        let flash = block_size * block_size * 4 * 2; // S and P blocks

        (naive, flash)
    }
}

include!("attention_async.rs");
include!("kv_scatter.rs");
include!("batch.rs");
include!("flash_decoding_graphed.rs");
include!("head_dim.rs");
