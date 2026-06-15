impl CudaExecutor {
    // ========================================================================
    // PAR-005: Cached GEMV Methods (avoid per-call weight transfers)
    // ========================================================================

    /// Execute Q4_K GEMV using cached weights - PAR-005
    ///
    /// Uses pre-uploaded weights from `quantized_weight_cache` to avoid
    /// CPU→GPU transfer on every forward pass. Weights must be loaded
    /// beforehand via `load_quantized_weights()`.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight tensor
    /// * `input` - Input vector (f32, length k)
    /// * `output` - Output vector (f32, length n)
    /// * `n` - Output dimension
    /// * `k` - Input dimension (must be divisible by 256)
    ///
    /// # Errors
    ///
    /// Returns error if weights not cached or kernel fails.
    pub fn q4k_gemv_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // PAR-057: Use TiledQ4KGemv for better performance (~4x fewer global reads)
        // Fall back to basic Q4KGemv if K not aligned to 256
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)
        let use_tiled = k.is_multiple_of(256) && k <= MAX_TILED_K;
        let use_chunked = k.is_multiple_of(256) && k > MAX_TILED_K;
        let outputs_per_block = 4u32;

        let (kernel_type, cache_key, config) = if use_chunked {
            // PAR-502: Use chunked kernel for large K dimensions (7B+ models)
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else if use_tiled {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            // NOTE: Shared memory is statically declared in PTX - do NOT pass dynamically
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else {
            let kt = KernelType::Q4KGemv { k, n };
            let ck = format!("q4k_gemv_{}_{}", k, n);
            let cfg = LaunchConfig::grid_2d(n, 1, 32, 1);
            (kt, ck, cfg)
        };

        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        // GH-215 FIX: Pad activations to ceil(K/256)*256 when K not 256-aligned.
        // The Q4K kernel reads activations at sb_idx*256+val_idx, which reaches
        // up to (num_super_blocks-1)*256+255. Without padding, this is an OOB read.
        let padded_k = ((k as usize + 255) / 256) * 256;
        let padded_input: std::borrow::Cow<'_, [f32]> = if padded_k > input.len() {
            let mut padded = vec![0.0f32; padded_k];
            padded[..input.len()].copy_from_slice(input);
            std::borrow::Cow::Owned(padded)
        } else {
            std::borrow::Cow::Borrowed(input)
        };

        // ALB-110: Use grow-only pooled buffers instead of per-call allocation.
        // Eliminates ~356K cuMemAlloc/cuMemFree per request that fragment the
        // CUDA allocator and crash the process after ~65 sustained completions.
        // Uses copy_*_at(offset=0) for partial copies into oversized pooled buffers.
        self.ensure_gemv_input_buffer(padded_k)?;
        self.ensure_gemv_output_buffer(n as usize)?;
        self.gemv_input_buffer
            .as_mut()
            .expect("just ensured")
            .copy_from_host_at(&padded_input, 0)?;

        let mut ptr_input = self
            .gemv_input_buffer
            .as_ref()
            .expect("just ensured")
            .as_ptr();
        let mut ptr_output = self
            .gemv_output_buffer
            .as_ref()
            .expect("just ensured")
            .as_ptr();
        let mut ptr_weights = weight_ptr;
        let mut k_val = k;
        let mut n_val = n;

        // Get module AFTER buffer setup to avoid overlapping &mut self borrows
        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

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

        self.stream.synchronize()?;
        self.gemv_output_buffer
            .as_ref()
            .expect("just ensured")
            .copy_to_host_at(output, 0)?;

        Ok(())
    }

    /// ALB-111: Upload input vector to GEMV input buffer.
    /// Call once before multiple GEMV launches with the same input.
    pub fn q4k_upload_to_input_buffer(&mut self, input: &[f32], k: u32) -> Result<(), GpuError> {
        let padded_k = ((k as usize + 255) / 256) * 256;
        let padded_input: std::borrow::Cow<'_, [f32]> = if padded_k > input.len() {
            let mut padded = vec![0.0f32; padded_k];
            padded[..input.len()].copy_from_slice(input);
            std::borrow::Cow::Owned(padded)
        } else {
            std::borrow::Cow::Borrowed(input)
        };
        self.ensure_gemv_input_buffer(padded_k)?;
        self.gemv_input_buffer
            .as_mut()
            .expect("just ensured")
            .copy_from_host_at(&padded_input, 0)?;
        Ok(())
    }

    /// ALB-111: Launch Q4K GEMV kernel with output to a specific device pointer.
    /// Input must already be in gemv_input_buffer (via q4k_upload_to_input_buffer).
    /// NO sync, NO D2H. Caller must sync and download.
    pub fn q4k_gemv_launch_to_ptr(
        &mut self,
        weight_name: &str,
        output_device_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        const MAX_TILED_K: u32 = 12_288;
        let use_tiled = k.is_multiple_of(256) && k <= MAX_TILED_K;
        let use_chunked = k.is_multiple_of(256) && k > MAX_TILED_K;
        let outputs_per_block = 4u32;

        let (kernel_type, cache_key, config) = if use_chunked {
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else if use_tiled {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else {
            let kt = KernelType::Q4KGemv { k, n };
            let ck = format!("q4k_gemv_{}_{}", k, n);
            let cfg = LaunchConfig::grid_2d(n, 1, 32, 1);
            (kt, ck, cfg)
        };

        let kernel_name = self.kernels.kernel_name(&kernel_type);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let mut ptr_input = self
            .gemv_input_buffer
            .as_ref()
            .expect("must upload first via q4k_upload_to_input_buffer")
            .as_ptr();
        let mut ptr_output = output_device_ptr;
        let mut ptr_weights = weight_ptr;
        let mut k_val = k;
        let mut n_val = n;

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

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

    /// ALB-111: Synchronize the CUDA stream.
    pub fn sync_stream(&mut self) -> Result<(), GpuError> {
        self.stream.synchronize()
    }

    /// ALB-111: Download from a GEMV output buffer by index.
    /// 0 = primary gemv_output_buffer, 1 = buffer B, 2 = buffer C.
    pub fn download_gemv_output(
        &self,
        buf_index: usize,
        output: &mut [f32],
    ) -> Result<(), GpuError> {
        let buf = match buf_index {
            0 => self
                .gemv_output_buffer
                .as_ref()
                .expect("buffer must exist"),
            1 => self
                .gemv_output_buffer_b
                .as_ref()
                .expect("buffer B must exist"),
            2 => self
                .gemv_output_buffer_c
                .as_ref()
                .expect("buffer C must exist"),
            _ => {
                return Err(GpuError::InvalidParameter(format!(
                    "Invalid buffer index: {buf_index}"
                )));
            },
        };
        buf.copy_to_host_at(output, 0)
    }

    /// PAR-023: Execute Q4_K GEMV with GPU buffer input/output (async, no sync)
    ///
    /// This is the async variant that keeps data on GPU. Used for pipelining
    /// multiple operations without CPU round-trips.
    ///
    /// # Arguments
    ///
    /// * `weight_name` - Name of cached weight buffer
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    ///
    /// # Returns
    ///
    /// GPU buffer containing output vector (not synchronized)
    pub fn q4k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        // PMAT-OXIDE-Q4K-001: cuda-oxide pre-generated PTX backend for Blackwell.
        // Opt-in (env APR_Q4K_OXIDE=1) AND compute capability >= 120 (sm_120+) AND
        // K aligned to 256. On sm_89 or opt-out, this returns None and we fall
        // through to the EXISTING TiledQ4KGemv path UNCHANGED (hot decode path).
        if let Some(result) = self.try_q4k_gemv_oxide_async(weight_ptr, input, n, k) {
            return result;
        }

        // CORRECTNESS-001: Use TiledQ4KGemv for aligned K (matches sync version)
        // The basic Q4KGemv kernel has the same scale extraction issue
        // PAR-502: sm_89 has 100KB shared memory limit, K * 4 bytes must fit
        const MAX_TILED_K: u32 = 12_288; // 48KB / 4 bytes = 12,288 floats (default static shared memory limit)
        let use_tiled = k.is_multiple_of(256) && k <= MAX_TILED_K;
        let use_chunked = k.is_multiple_of(256) && k > MAX_TILED_K;
        let outputs_per_block = 4u32;

        let (kernel_type, cache_key, config) = if use_chunked {
            // PAR-502: Use chunked kernel for large K dimensions (7B+ models)
            let kt = KernelType::ChunkedTiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("chunked_tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else if use_tiled {
            let kt = KernelType::TiledQ4KGemv {
                k,
                n,
                outputs_per_block,
            };
            let ck = format!("tiled_q4k_gemv_{}_{}_{}", k, n, outputs_per_block);
            let num_blocks = (n + outputs_per_block - 1) / outputs_per_block;
            let cfg = LaunchConfig::grid_2d(num_blocks, 1, 128, 1);
            (kt, ck, cfg)
        } else {
            let kt = KernelType::Q4KGemv { k, n };
            let ck = format!("q4k_gemv_{}_{}", k, n);
            let cfg = LaunchConfig::grid_2d(n, 1, 32, 1);
            (kt, ck, cfg)
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

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let mut ptr_output = buf_output.as_ptr();
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

        // PAR-023: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-058: Execute Q6_K GEMV using cached weight (async, no sync)
    ///
    /// Same as q4k_gemv_cached_async but for Q6_K quantized weights.
    /// Used for LM head when it's Q6K quantized.
    pub fn q6k_gemv_cached_async(
        &mut self,
        weight_name: &str,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Get cached weight buffer (ALB-098: checks pool first, then individual cache)
        let weight_ptr = self.get_quantized_weight_ptr(weight_name)?;

        let use_mwv = self.gpu_profile.q6k != crate::cuda::gpu_profile::Q6kVariant::Legacy && k.is_multiple_of(256);
        let num_warps = self.gpu_profile.mwv_warps;

        let (kernel_type, cache_key, config) = if use_mwv {
            let kt = KernelType::MwvQ6KGemv { k, n, num_warps };
            let ck = format!("mwv_q6k_gemv_{}_{}_{}", k, n, num_warps);
            let cfg = LaunchConfig::grid_2d(n, 1, num_warps * 32, 1);
            (kt, ck, cfg)
        } else {
            let kt = KernelType::Q6KGemv { k, n };
            let ck = format!("q6k_gemv_{}_{}", k, n);
            let cfg = LaunchConfig::grid_2d(n, 1, 32, 1);
            (kt, ck, cfg)
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

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        let mut ptr_output = buf_output.as_ptr();
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

        // PAR-058: NO synchronization here - caller can chain operations
        Ok(buf_output)
    }

    /// PAR-043: Execute Q4_K GEMV using pre-indexed device pointer (async, no sync)
    ///
    /// This eliminates HashMap lookup + string formatting overhead (~10ms per token).
    /// Weight pointer must be from `indexed_layer_weights` populated by `build_indexed_weights()`.
    ///
    /// # Arguments
    ///
    /// * `weight_ptr` - Raw device pointer to Q4K weight data
    /// * `input` - GPU buffer containing input vector
    /// * `n` - Output dimension
    /// * `k` - Input dimension
    #[inline]
    pub fn q4k_gemv_indexed_async(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Validate pointer before kernel launch — launching with ptr=0
        // crashes the kernel and permanently poisons the CUDA context.
        if weight_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "null weight pointer in q4k_gemv_indexed_async".to_string(),
            ));
        }

        // Allocate output buffer
        let buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;

        // PAR-082-V2: Use MwvQ4KGemv with configurable warp count
        let num_warps = self.gpu_profile.mwv_warps;
        let kernel_type = KernelType::MwvQ4KGemv { k, n, num_warps };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("mwv_q4k_gemv_{}_{}_{}", k, n, num_warps);

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // num_warps * 32 threads per output element
        let threads = num_warps * 32;
        let config = LaunchConfig::grid_2d(n, 1, threads, 1);
        let mut ptr_output = buf_output.as_ptr();
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

        Ok(buf_output)
    }

    /// PMAT-734: minimum K for the oxide Q4K backend to be a net throughput win.
    ///
    /// The oxide kernel uses a fixed 32-threads/row reduction. Its advantage over
    /// the tiled shared-memory kernel GROWS with K — each thread processes K/32
    /// super-block contributions, so a larger K amortizes the kernel's launch +
    /// atomic-reduction overhead. The tiled kernel instead stages the activation
    /// vector in shared memory, which keeps it competitive (or ahead) at small-K
    /// where the oxide kernel's per-row reduction is launch-overhead bound.
    ///
    /// MEASURED crossover on gx10 (GB10 Blackwell sm_121, cc=121, driver
    /// 590.48.01, CUDA 13.0), median of 7 batches x 200 async launches after 50
    /// warmup, weights cached on GPU + input device-resident (pure launch timing
    /// on the production async path). K-sweep at fixed N=1536, two runs (the
    /// us/launch below are run 1; the two-run speedup band is shown):
    ///
    /// | N    | K    | oxide us | tiled us | speedup (run1/run2) | verdict       |
    /// |------|------|----------|----------|---------------------|---------------|
    /// | 4096 | 2048 | 102.6    | 103.9    | 1.01x (attn shape)  | tie/LOSE      |
    /// | 1536 | 4096 |  85.1    |  81.1    | 0.95x / 1.02x       | tie/LOSE      |
    /// | 1536 | 5120 | 101.0    | 106.2    | 1.05x / 1.01x       | tie           |
    /// | 1536 | 6144 | 116.5    | 114.7    | 0.99x / 1.06x       | tie           |
    /// | 1536 | 6656 | 122.3    | 169.1    | 1.38x / 1.47x       | WIN (crossover)|
    /// | 1536 | 7168 | 131.3    | 186.7    | 1.42x / 1.55x       | WIN           |
    /// | 1536 | 8192 | 147.5    | 206.7    | 1.40x / 1.50x       | WIN           |
    /// | 1536 | 8960 | 158.9    | 304.8    | 1.92x / 2.05x       | BEAT (~2.0x)  |
    ///
    /// The crossover is at K=6656: below it oxide only ties tiled (~1.0-1.06x,
    /// NOT a beat); at and above K=6656 oxide is a clear >=1.38x win, climbing to
    /// ~2.0x at K=8960. (NB: the earlier "FFN wins at all K" reading was an
    /// artifact of comparing the N=1536 FFN win to the N=4096 attention loss —
    /// the N=1536 K-sweep shows oxide does NOT win until K=6656.)
    ///
    /// We set the threshold CONSERVATIVELY at `K >= 6656`: the lowest swept K
    /// where BOTH runs clear the 1.25x beat threshold. Common FFN down/up/gate K
    /// dims (1.5B: 8960; 7B: 11008/14336) clear it and route to oxide; common
    /// attention K dims (1536/2048) and the marginal mid-K region (<=6144) stay
    /// on the proven TiledQ4KGemv path. This makes the `APR_Q4K_OXIDE=1` opt-in a
    /// genuine net win: FFN faster, attention unchanged. See the throughput gate
    /// (`q4k_gemv_oxide_throughput.rs`) and `contracts/beat-q4k-oxide-sm121-v1.yaml`.
    const OXIDE_MIN_K: u32 = 6656;

    /// PMAT-734: K-based shape gate predicate for the oxide Q4K backend.
    ///
    /// Returns `true` iff `k` is a shape where the oxide kernel is a measured
    /// throughput WIN over `TiledQ4KGemv` on Blackwell sm_121: `k` must be Q4K
    /// super-block aligned (`k % 256 == 0`) AND large enough (`k >= OXIDE_MIN_K`)
    /// to clear the measured crossover. Small-K attention GEMVs return `false`
    /// and stay on the tiled path. Pure (no device state) so it is unit-testable
    /// without GPU hardware. See [`Self::OXIDE_MIN_K`] for the measured crossover.
    #[inline]
    #[must_use]
    pub(crate) fn oxide_k_shape_gate_passes(k: u32) -> bool {
        k.is_multiple_of(256) && k >= Self::OXIDE_MIN_K
    }

    /// PMAT-734: the oxide Q4K K shape-gate threshold (see [`Self::OXIDE_MIN_K`]).
    /// Exposed so the throughput gate can read the same value the dispatch uses,
    /// keeping the test threshold and the production threshold from drifting.
    #[inline]
    #[must_use]
    pub(crate) fn oxide_min_k() -> u32 {
        Self::OXIDE_MIN_K
    }

    /// PMAT-OXIDE-Q4K-001: pre-generated cuda-oxide Q4K dequant-matvec backend.
    ///
    /// Returns `Some(result)` only when the oxide path is selected:
    /// - opt-in via env `APR_Q4K_OXIDE=1` (default OFF — nothing changes without it),
    /// - device compute capability >= 120 (Blackwell sm_120+),
    /// - `k` is a multiple of 256 (Q4K super-block alignment), and
    /// - `k >= OXIDE_MIN_K` (large-K FFN-class shapes where oxide wins — small-K
    ///   attention shapes stay on the tiled kernel; see [`Self::OXIDE_MIN_K`]).
    ///
    /// On sm_89 (RTX 4090, the proven hot path), when opt-out, OR for small-K
    /// attention GEMVs, returns `None` so the caller falls through to the
    /// EXISTING `TiledQ4KGemv` hand-PTX path, which is left byte-for-byte
    /// unchanged.
    ///
    /// The kernel uses the verified ABI `(data: *const u8, x: *const f32,
    /// y: *mut f32, m: u32, k: u32, t: u32)` with `t = 32` threads/row. `y` MUST
    /// be zeroed before launch because the kernel accumulates via
    /// `atom.global.add.f32`. Launch is `total = m * t`, block = 256,
    /// grid = ceil(total / 256). `m` is the output dim (`n` in this method's
    /// naming).
    ///
    /// The pre-generated PTX (`.target sm_121`, `.version 8.8`) is embedded as a
    /// static asset and loaded once through the existing `compile_ptx` / cubin
    /// cache path — no cuda-oxide build dependency.
    fn try_q4k_gemv_oxide_async(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Option<Result<GpuBuffer<f32>, GpuError>> {
        // Gate 1: explicit opt-in. Default OFF → return None, no behavior change.
        if std::env::var("APR_Q4K_OXIDE").as_deref() != Ok("1") {
            return None;
        }
        // Gate 2: Blackwell sm_120+ only. The asset is `.target sm_121` and only
        // wins / dodges GH-480 there. Preserve the sm_89 fallback exactly.
        if self.gpu_profile.cc < 120 {
            return None;
        }
        // Gate 3 + 4 (PMAT-734): Q4K super-block alignment (K % 256 == 0) AND the
        // K-based SHAPE GATE. The throughput gate measured the oxide kernel only
        // BEATS TiledQ4KGemv at large K: it ties (~1.0x) up to K=6144 and wins
        // >=1.38x from K=6656 (~2.0x at K=8960), while small-K attention GEMVs
        // (K=2048) tie/lose. Route to oxide ONLY at or above the conservative
        // crossover threshold so the opt-in is a net win (FFN faster, attention
        // unchanged). Below the threshold → None → existing TiledQ4KGemv path.
        // See [`Self::OXIDE_MIN_K`] for the measured per-K crossover table.
        if !Self::oxide_k_shape_gate_passes(k) {
            return None;
        }

        Some(self.q4k_gemv_oxide_async_inner(weight_ptr, input, n, k))
    }

    /// Inner implementation of the oxide backend (split out so the gated entry
    /// point stays trivially simple). See [`Self::try_q4k_gemv_oxide_async`].
    pub(crate) fn q4k_gemv_oxide_async_inner(
        &mut self,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Validate weight pointer — launching with ptr=0 poisons the context.
        if weight_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "null weight pointer in q4k_gemv_oxide_async".to_string(),
            ));
        }

        // The PTX is dimension-independent (dims are runtime args), so a single
        // cached module serves all (k, n). Cache key is stable.
        const OXIDE_CACHE_KEY: &str = "oxide_q4k_matvec_sm121";
        const OXIDE_ENTRY: &str = "q4k_matvec";
        const OXIDE_PTX: &str = include_str!("../ptx/q4k_matvec_oxide.sm121.ptx");

        if !self.modules.contains_key(OXIDE_CACHE_KEY) {
            let module = self.compile_ptx(OXIDE_PTX)?;
            self.modules.insert(OXIDE_CACHE_KEY.to_string(), module);
        }
        let module = self
            .modules
            .get_mut(OXIDE_CACHE_KEY)
            .expect("oxide module just inserted");

        // Allocate output buffer and ZERO it — the kernel accumulates with
        // atom.global.add.f32, so a non-zeroed y produces garbage.
        let mut buf_output = GpuBuffer::<f32>::new(&self.context, n as usize)?;
        // Zero via the compute stream. `&self.stream` derefs PoolableStream
        // (Deref<Target = CudaStream>) to the &CudaStream zero_async expects.
        buf_output.zero_async(&self.stream)?;

        // ABI: (data, x, y, m, k, t). t = 32 threads per output row.
        let mut ptr_weights = weight_ptr; // data: *const u8
        let mut ptr_input = input.as_ptr(); // x: *const f32
        let mut ptr_output = buf_output.as_ptr(); // y: *mut f32
        let mut m_val = n; // m: output rows
        let mut k_val = k; // k: input dim
        let mut t_val: u32 = 32; // t: threads/row (fixed)

        // Launch: total = m * t, block = 256, grid = ceil(total / 256).
        let total = m_val.saturating_mul(t_val);
        let num_blocks = total.div_ceil(256);
        let config = LaunchConfig::grid_2d(num_blocks, 1, 256, 1);

        // SAFETY: pointers are valid device buffers, arg order matches the
        // verified `q4k_matvec` ABI, and `y` was zeroed above for the atomic add.
        unsafe {
            self.stream.launch_kernel(
                module,
                OXIDE_ENTRY,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_weights) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_input) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_output) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut t_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // NO synchronization here — caller can chain operations (matches the
        // existing async path contract).
        Ok(buf_output)
    }

    /// PMAT-OXIDE-Q4K-001: synchronous host-in/host-out Q4K GEMV via the
    /// cuda-oxide PTX backend. Mirrors [`Self::q4k_gemv`] but forces the oxide
    /// kernel.
    ///
    /// This is the parity/bench entry point: it uploads `weight` + `input`,
    /// runs the embedded `q4k_matvec` kernel (with `y` zeroed for the atomic
    /// accumulation), synchronizes, and copies the result back. It deliberately
    /// does NOT consult `APR_Q4K_OXIDE` so tests/benches are deterministic; it
    /// only requires Blackwell (cc >= 120) since the asset is `.target sm_121`.
    ///
    /// # Errors
    /// Returns `Err` if the device is not sm_120+ (the asset won't load on
    /// sm_89), if `k` is not a multiple of 256, or on any CUDA failure.
    pub fn q4k_gemv_oxide(
        &mut self,
        weight: &[u8],
        input: &[f32],
        output: &mut [f32],
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        if self.gpu_profile.cc < 120 {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "oxide Q4K backend requires compute capability >= 120 (sm_120+), \
                 device is cc={}",
                self.gpu_profile.cc
            )));
        }
        if !k.is_multiple_of(256) {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "oxide Q4K backend requires k % 256 == 0, got k={k}"
            )));
        }

        // Upload weight bytes to a fresh device buffer (test/bench path; the
        // production path uses the cached weight pointer).
        let buf_weights = GpuBuffer::from_host(&self.context, weight)?;

        // GH-215 style padding: the kernel reads activations up to the padded
        // super-block boundary; pad to ceil(k/256)*256 to avoid OOB reads.
        let padded_k = ((k as usize + 255) / 256) * 256;
        let padded_input: std::borrow::Cow<'_, [f32]> = if padded_k > input.len() {
            let mut padded = vec![0.0f32; padded_k];
            padded[..input.len()].copy_from_slice(input);
            std::borrow::Cow::Owned(padded)
        } else {
            std::borrow::Cow::Borrowed(input)
        };
        let buf_input = GpuBuffer::from_host(&self.context, &padded_input)?;

        let buf_output = self.q4k_gemv_oxide_async_inner(buf_weights.as_ptr(), &buf_input, n, k)?;
        self.stream.synchronize()?;
        buf_output.copy_to_host(output)?;

        // Keep the weight/input buffers alive until after the launch+sync.
        drop(buf_weights);
        drop(buf_input);
        Ok(())
    }
}
