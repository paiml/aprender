impl CudaExecutor {

    /// Execute a tiled GEMM kernel: C = A @ B
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (m x k, row-major)
    /// * `b` - Input matrix B (k x n, row-major)
    /// * `c` - Output matrix C (m x n, row-major)
    /// * `m` - Number of rows in A and C
    /// * `n` - Number of columns in B and C
    /// * `k` - Number of columns in A / rows in B
    ///
    /// # Errors
    ///
    /// Returns error if kernel execution fails.
    pub fn gemm(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
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

        // Generate PTX for this configuration
        // PARITY-003: Enable simpler Gemv (warp-reduce) for M=1 operations
        let use_gemv = m == 1;
        let (kernel_type, cache_key) = if use_gemv {
            (KernelType::Gemv { k, n }, format!("gemv_{}_{}", k, n))
        } else {
            (
                KernelType::GemmTiled {
                    m,
                    n,
                    k,
                    tile_size: 32,
                },
                format!("gemm_{}_{}_{}_{}", m, n, k, 32),
            )
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);

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

        // Launch configuration differs for Gemv vs GEMM
        // PARITY-003: Enable simpler Gemv with correct config
        let config = if use_gemv {
            // Simple Gemv: 32 threads (one warp) per block, N blocks
            // Each block computes one output element y[block_id]
            LaunchConfig::grid_2d(n, 1, 32, 1)
        } else {
            // GEMM: 2D grid of 32x32 tiles
            // PARITY-114 FIX: Grid X is for columns (N), Grid Y is for rows (M)
            LaunchConfig::grid_2d(
                (n + 31) / 32, // Grid X - columns (N dimension)
                (m + 31) / 32, // Grid Y - rows (M dimension)
                32,            // Block X
                32,            // Block Y
            )
        };

        // Get raw pointers for kernel args
        let mut ptr_a = buf_a.as_ptr();
        let mut ptr_b = buf_b.as_ptr();
        let mut ptr_c = buf_c.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // Launch kernel
        // SAFETY: Buffers are valid, config matches kernel expectations
        // PARITY-003: Enable GEMV for M=1 operations
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            if use_gemv {
                // GEMV kernel: y = B * x where x is A (1×K row as K vector), B is K×N, y is C (1×N as N vector)
                // Args: y_ptr, a_ptr (matrix), x_ptr, k_dim, n_dim
                self.stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void, // y_ptr (output)
                        std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void, // a_ptr (K×N matrix)
                        std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void, // x_ptr (K input vector)
                        std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void, // k_dim
                        std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void, // n_dim
                    ],
                )?;
            } else {
                // GEMM kernel: C = A × B
                // Args: a_ptr, b_ptr, c_ptr, m, n, k
                // GH-282: Keep as u32 to match kernel .param .u32 declarations
                let mut m_val = m;
                let mut n_val_i32 = n;
                let mut k_val_i32 = k;
                self.stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut n_val_i32) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_val_i32) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Synchronize and copy result back
        self.stream.synchronize()?;
        buf_c.copy_to_host(c)?;

        Ok(())
    }

    /// Execute GEMV using cached weight matrix (PARITY-120: 10x speedup)
    ///
    /// This is the fast path for single-token generation (M=1).
    /// The weight matrix must be pre-loaded via `load_weights()`.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight matrix
    /// * `x` - Input vector (K elements)
    /// * `y` - Output vector (N elements)
    /// * `k` - Input dimension
    /// * `n` - Output dimension
    ///
    /// # Errors
    ///
    /// Returns error if weight not cached or kernel execution fails.
    pub fn gemv_cached(
        &mut self,
        weight_name: &str,
        x: &[f32],
        y: &mut [f32],
        k: u32,
        n: u32,
    ) -> Result<(), GpuError> {
        // Validate sizes
        if x.len() != k as usize {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "GEMV input size mismatch: got {}, expected {}",
                x.len(),
                k
            )));
        }
        if y.len() != n as usize {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "GEMV output size mismatch: got {}, expected {}",
                y.len(),
                n
            )));
        }

        // Get cached weight buffer
        let buf_w = self.weight_cache.get(weight_name).ok_or_else(|| {
            GpuError::InvalidLaunchConfig(format!("Weight '{}' not cached on GPU", weight_name))
        })?;

        // PARITY-003: Use simpler Gemv kernel (32 threads warp-reduce) instead of CoalescedGemv
        let kernel_type = KernelType::Gemv { k, n };
        let cache_key = format!("gemv_simple_{}_{}", k, n);
        let kernel_name = self.kernels.kernel_name(&kernel_type);

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

        // Allocate only input/output buffers (weight stays on GPU!)
        let buf_x = GpuBuffer::from_host(&self.context, x)?;
        let y_zeros = vec![0.0f32; n as usize];
        let buf_y = GpuBuffer::from_host(&self.context, &y_zeros)?;

        // PARITY-003: Simple Gemv config - 32 threads (one warp) per block, N blocks
        let config = LaunchConfig::grid_2d(n, 1, 32, 1);

        // Get raw pointers
        let mut ptr_y = buf_y.as_ptr();
        let mut ptr_w = buf_w.as_ptr();
        let mut ptr_x = buf_x.as_ptr();
        let mut k_val = k;
        let mut n_val = n;

        // Launch kernel: y = W * x
        // SAFETY: Buffers are valid, config matches kernel expectations
        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_y) as *mut std::ffi::c_void, // y_ptr (output)
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void, // w_ptr (K×N matrix, CACHED)
                    std::ptr::from_mut(&mut ptr_x) as *mut std::ffi::c_void, // x_ptr (K input vector)
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void, // k_dim
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void, // n_dim
                ],
            )?;
        }

        // Synchronize and copy result back
        self.stream.synchronize()?;
        buf_y.copy_to_host(y)?;

        Ok(())
    }

    /// Execute optimized GEMM kernel (IMP-900a)
    ///
    /// Uses larger tile sizes and register blocking for better performance.
    /// Provides ~2-3x improvement over naive tiled GEMM.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (m × k)
    /// * `b` - Input matrix B (k × n)
    /// * `c` - Output matrix C (m × n)
    /// * `m` - Number of rows in A and C
    /// * `n` - Number of columns in B and C
    /// * `k` - Number of columns in A / rows in B
    /// * `tile_size` - Tile size for shared memory (32 or 64)
    ///
    /// # Errors
    ///
    /// Returns error if kernel execution fails.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_optimized(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: u32,
        n: u32,
        k: u32,
        tile_size: u32,
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

        // IMP-900a: Use optimized kernel with larger tiles
        let reg_block = if tile_size >= 64 { 8 } else { 4 };
        let kernel_type = KernelType::GemmOptimized {
            m,
            n,
            k,
            tile_size,
            reg_block,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("gemm_opt_{}_{}_{}_{}", m, n, k, tile_size);

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

        // Launch configuration with optimized tile size
        // PARITY-114 FIX: Grid X is for columns (N), Grid Y is for rows (M)
        let config = LaunchConfig::grid_2d(
            (n + tile_size - 1) / tile_size, // Grid X - columns (N dimension)
            (m + tile_size - 1) / tile_size, // Grid Y - rows (M dimension)
            tile_size,                       // Block X
            tile_size,                       // Block Y
        );

        // Get raw pointers for kernel args
        let mut ptr_a = buf_a.as_ptr();
        let mut ptr_b = buf_b.as_ptr();
        let mut ptr_c = buf_c.as_ptr();
        // GH-282: Keep as u32 to match kernel .param .u32 declarations
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

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
}
