//! GEMM dispatch for cuBLAS prefill.
//!
//! Contains FP8 GEMM, dequant launches, HGEMM, WMMA GEMM, fused Q4K GEMM,
//! and the top-level cublas_prefill_gemm dispatch.

use super::super::super::*;

impl CudaExecutor {
    /// PMAT-053b: Get cached FP8 E4M3 weight with per-tensor scaling.
    ///
    /// On cache miss: dequant Q4K/Q6K → FP32 → absmax → scaled FP8 E4M3 → cache.
    /// Also stores the dequant scale (absmax/448) in fp8_weight_scales for cuBLASLt.
    pub(crate) fn get_or_cache_fp8_weight(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<u64, GpuError> {
        if let Some(buf) = self.fp8_weight_cache.get(&weight_ptr) {
            return Ok(buf.as_ptr());
        }

        // Cache miss: dequant → FP32 scratch
        let f32_ptr = match qtype {
            WeightQuantType::Q4K => self.dequant_q4k_to_scratch(weight_ptr, n, k)?,
            WeightQuantType::Q6K => self.dequant_q6k_to_scratch(weight_ptr, n, k)?,
            _ => {
                return Err(GpuError::InvalidParameter(format!(
                    "get_or_cache_fp8_weight: unsupported qtype {:?}",
                    qtype
                )))
            },
        };

        let count = n as usize * k as usize;

        // PMAT-053b: Compute per-tensor absmax for scaling
        let absmax = self.gpu_absmax(f32_ptr, count as u32)?;
        let absmax = if absmax == 0.0 { 1.0 } else { absmax };
        let quant_scale = 448.0 / absmax;
        let dequant_scale = absmax / 448.0;

        // Allocate persistent FP8 buffer [N × K] — 1 byte per element
        let fp8_buf = GpuBuffer::<u8>::new(&self.context, count)?;
        let fp8_ptr = fp8_buf.as_ptr();

        // Convert FP32 → scaled FP8 E4M3
        self.convert_f32_to_e4m3_scaled(f32_ptr, fp8_ptr, count as u32, quant_scale)?;

        // Store dequant scale as CPU float — used as GEMM alpha (constant, no sync needed)
        self.fp8_weight_scales.insert(weight_ptr, dequant_scale);

        self.fp8_weight_cache.insert(weight_ptr, fp8_buf);
        Ok(fp8_ptr)
    }

    /// PMAT-079: Fully async FP8 E4M3 GEMM — zero CPU syncs.
    ///
    /// Pipeline (all on device, no CPU readback):
    ///   1. absmax_reduce → device absmax_buf (no sync)
    ///   2. f32_to_e4m3_device_scaled → reads absmax from device, writes FP8 + act_dequant
    ///   3. gemm_fp8_e4m3_to_f16 → unscaled GEMM with alpha=1.0
    ///   4. f16_to_f32_device_scaled → reads act_dequant × weight_dequant from device
    ///
    /// The GEMM computes raw FP8 dot products (no scaling). The dequant is applied
    /// during the FP16→FP32 conversion: output = f16_val × (act_absmax/448) × (w_absmax/448).
    /// This avoids both the GPU→CPU absmax sync AND cuBLASLt scale pointer issues.
    #[allow(clippy::too_many_arguments)]
    fn cublas_prefill_fp8_gemm(
        &mut self,
        w_fp8_ptr: u64,
        weight_key: u64, // original weight_ptr used as key into fp8_weight_scales
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32, // sequence/batch length (tokens)
        n: u32, // output dimension
        k: u32, // input dimension
    ) -> Result<(), GpuError> {
        let detail_trace = std::env::var("PREFILL_DETAIL_TRACE").is_ok();
        let t0 = if detail_trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // cuBLASLt FP8 requires batch dimension aligned to 16
        let m_padded = (m + 15) & !15;

        // Step 1+2: Device-side absmax + FP8 conversion (zero CPU syncs)
        // PMAT-084: Cache FP8 activation — skip redundant absmax+convert when
        // multiple GEMMs share the same input (QKV phase, FFN gate+up).
        // Saves 84 kernel pairs per prefill (3 per layer × 28 layers).
        let input_actual_count = (m as usize * k as usize) as u32;
        let input_padded_count = m_padded as usize * k as usize;
        self.ensure_fp8_activation_scratch(input_padded_count)?;
        let input_fp8_ptr = self
            .fp8_activation_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();

        // Ensure persistent dequant buffer exists
        if self.fp8_act_dequant_buf.is_none() {
            self.fp8_act_dequant_buf = Some(GpuBuffer::<f32>::new(&self.context, 1)?);
        }
        let act_dequant_ptr = self
            .fp8_act_dequant_buf
            .as_ref()
            .expect("just allocated")
            .as_ptr();

        let cache_key = (packed_input_ptr, input_actual_count);
        if self.fp8_activation_cache_key == Some(cache_key) {
            // PMAT-084: Reuse cached FP8 activation + dequant scale.
            // QKV phase: Q computes, K+V reuse. FFN: gate computes, up reuses.
            // 3 hits/layer × 28 layers = 84 saved absmax+convert pairs.
            if detail_trace {
                eprintln!("[PMAT-084] FP8 activation cache HIT ptr={packed_input_ptr:#x} count={input_actual_count}");
            }
        } else {
            let absmax_ptr = self.gpu_absmax_device(packed_input_ptr, input_actual_count)?;
            self.convert_f32_to_e4m3_device_scaled(
                packed_input_ptr,
                input_fp8_ptr,
                input_actual_count,
                absmax_ptr,
                act_dequant_ptr,
            )?;
            self.fp8_activation_cache_key = Some(cache_key);
        }

        // Look up weight dequant scale (CPU float, constant per weight, no sync needed)
        let weight_dequant = *self.fp8_weight_scales.get(&weight_key).ok_or_else(|| {
            GpuError::InvalidParameter(format!(
                "FP8 weight scale not found for key {weight_key:#x}"
            ))
        })?;

        let t1 = if detail_trace {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Step 3: cuBLASLt FP8 GEMM with alpha=weight_dequant → FP16 output
        // weight_dequant is a constant CPU float (computed once at weight cache time).
        // This partially dequants: D = (w_max/448) × FP8(A) × FP8(B)
        // = (448/act_max) × true_result. The act_dequant (act_max/448) is applied in step 4.
        let output_padded_count = n as usize * m_padded as usize;
        self.ensure_fp16_activation_scratch(output_padded_count)?;
        let f16_output_ptr = self
            .fp16_activation_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();

        if self.cublaslt_handle.is_none() {
            self.cublaslt_handle = Some(trueno_gpu::driver::CublasLtHandle::new()?);
        }
        // PMAT-086: Use cached GEMM to avoid per-call descriptor creation.
        // 168 GEMMs per prefill × ~30μs descriptor overhead = ~5ms savings.
        let lt_handle = self.cublaslt_handle.as_mut().expect("just created");
        lt_handle.gemm_fp8_e4m3_to_f16_cached(
            n as i32,
            m_padded as i32,
            k as i32,
            weight_dequant, // alpha = w_absmax/448 (constant, no sync needed)
            w_fp8_ptr,
            k as i32,
            input_fp8_ptr,
            k as i32,
            0.0,
            f16_output_ptr,
            n as i32,
            &self.stream,
        )?;

        let t2 = if detail_trace {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Step 4: Convert FP16→FP32 with device-side act_dequant scaling.
        // Reads act_dequant (act_absmax/448) from device, multiplies each element by it.
        // Combined with step 3 alpha: D_f32 = f16_val × act_dequant = true_result.
        let output_actual_count = n as usize * m as usize;
        self.convert_f16_to_f32_act_scaled(
            f16_output_ptr,
            packed_output_ptr,
            output_actual_count as u32,
            act_dequant_ptr,
        )?;

        if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
            self.stream.synchronize()?;
            let t3 = std::time::Instant::now();
            eprintln!(
                "[FP8-TRACE] M={} (pad={}) N={} K={}: absmax+convert={:.3}ms gemm={:.3}ms f16->f32+scale={:.3}ms total={:.3}ms",
                m,
                m_padded,
                n,
                k,
                t1.duration_since(t0).as_secs_f64() * 1000.0,
                t2.duration_since(t1).as_secs_f64() * 1000.0,
                t3.duration_since(t2).as_secs_f64() * 1000.0,
                t3.duration_since(t0).as_secs_f64() * 1000.0,
            );
        }

        Ok(())
    }

    /// Launch Q4K dequant kernel to an arbitrary output buffer
    fn launch_dequant_q4k(
        &mut self,
        weight_ptr: u64,
        output_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let num_sb = (k + 255) / 256;
        let cache_key = format!("q4k_dequant_{k}_{n}");
        if !self.modules.contains_key(&cache_key) {
            let kernel_type = KernelType::Q4KDequant { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");
        let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

        let mut ptr_out = output_ptr;
        let mut ptr_w = weight_ptr;
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All pointers are valid GPU allocations, dimensions verified by caller
        unsafe {
            self.stream.launch_kernel(
                module,
                "q4k_dequant_to_f32",
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-065: Launch Q4K → FP16 direct dequant kernel
    ///
    /// Dequants Q4K super-blocks directly to FP16 output (no F32 intermediate).
    /// Half the output bandwidth of launch_dequant_q4k (2 B/elem vs 4 B/elem).
    fn launch_dequant_q4k_fp16(
        &mut self,
        weight_ptr: u64,
        output_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let num_sb = (k + 255) / 256;
        let cache_key = format!("q4k_dequant_fp16_{k}_{n}");
        if !self.modules.contains_key(&cache_key) {
            let kernel_type = KernelType::Q4KDequantFp16 { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");
        let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

        let mut ptr_out = output_ptr;
        let mut ptr_w = weight_ptr;
        let mut k_val = k;
        let mut n_val = n;

        unsafe {
            self.stream.launch_kernel(
                module,
                "q4k_dequant_to_f16",
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-065: Dequant Q4K → FP16 temp buffer for L2-cached HGEMM
    ///
    /// Per-matmul dequant: Q4K weights (DRAM) → FP16 temp (L2-hot) → cuBLAS HGEMM.
    /// Reads 3.56x less from DRAM vs cached FP16 HGEMM (0.5625 vs 2.0 B/elem).
    /// The FP16 temp buffer (≤27.5 MB for largest matrix) fits in RTX 4060's 32 MB L2,
    /// so cuBLAS reads from L2 instead of DRAM.
    ///
    /// Uses a separate `fp16_dequant_temp` buffer (not `fp16_activation_scratch`,
    /// which is used for input activation conversion in cublas_prefill_hgemm).
    fn dequant_q4k_fp16_temp(&mut self, weight_ptr: u64, n: u32, k: u32) -> Result<u64, GpuError> {
        let count = n as usize * k as usize;
        // Ensure temp buffer is large enough
        let need_alloc = match &self.fp16_dequant_temp {
            Some(buf) => buf.len() < count,
            None => true,
        };
        if need_alloc {
            self.fp16_dequant_temp = Some(GpuBuffer::<u16>::new(&self.context, count)?);
        }
        let fp16_ptr = self
            .fp16_dequant_temp
            .as_ref()
            .expect("temp just allocated")
            .as_ptr();
        self.launch_dequant_q4k_fp16(weight_ptr, fp16_ptr, n, k)?;
        Ok(fp16_ptr)
    }

    /// Launch Q6K dequant kernel to an arbitrary output buffer
    fn launch_dequant_q6k(
        &mut self,
        weight_ptr: u64,
        output_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let num_sb = (k + 255) / 256;
        let cache_key = format!("q6k_dequant_{k}_{n}");
        if !self.modules.contains_key(&cache_key) {
            let kernel_type = KernelType::Q6KDequant { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");
        let config = LaunchConfig::grid_2d(n, num_sb, 32, 1);

        let mut ptr_out = output_ptr;
        let mut ptr_w = weight_ptr;
        let mut k_val = k;
        let mut n_val = n;

        // SAFETY: All pointers are valid GPU allocations, dimensions verified by caller
        unsafe {
            self.stream.launch_kernel(
                module,
                "q6k_dequant_to_f32",
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// Dequantize Q4K weights on GPU into FP32 scratch buffer
    fn dequant_q4k_to_scratch(&mut self, weight_ptr: u64, n: u32, k: u32) -> Result<u64, GpuError> {
        self.ensure_dequant_scratch(n, k)?;
        let scratch_ptr = self
            .dequant_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();
        self.launch_dequant_q4k(weight_ptr, scratch_ptr, n, k)?;
        Ok(scratch_ptr)
    }

    /// PMAT-026: Dequantize Q6K weights on GPU into FP32 scratch buffer
    fn dequant_q6k_to_scratch(&mut self, weight_ptr: u64, n: u32, k: u32) -> Result<u64, GpuError> {
        self.ensure_dequant_scratch(n, k)?;
        let scratch_ptr = self
            .dequant_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();
        self.launch_dequant_q6k(weight_ptr, scratch_ptr, n, k)?;
        Ok(scratch_ptr)
    }

    /// PMAT-031: Get cached FP16 weight or dequant+convert+cache on first access.
    ///
    /// On cache miss: dequant Q4K/Q6K → FP32 scratch → convert to FP16 → cache.
    /// On cache hit: return cached FP16 pointer directly (zero dequant cost).
    pub(crate) fn get_or_cache_fp16_weight(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<u64, GpuError> {
        if let Some(buf) = self.fp16_weight_cache.get(&weight_ptr) {
            return Ok(buf.as_ptr());
        }

        // Cache miss: dequant → FP32 scratch
        let f32_ptr = match qtype {
            WeightQuantType::Q4K => self.dequant_q4k_to_scratch(weight_ptr, n, k)?,
            WeightQuantType::Q6K => self.dequant_q6k_to_scratch(weight_ptr, n, k)?,
            _ => {
                return Err(GpuError::InvalidParameter(format!(
                    "get_or_cache_fp16_weight: unsupported qtype {:?}",
                    qtype
                )))
            },
        };

        // Allocate persistent FP16 buffer [N × K]
        let count = n as usize * k as usize;
        let fp16_buf = GpuBuffer::<u16>::new(&self.context, count)?;
        let fp16_ptr = fp16_buf.as_ptr();

        // Convert FP32 → FP16 (same stream, ordered after dequant)
        self.convert_f32_to_f16(f32_ptr, fp16_ptr, count as u32)?;

        self.fp16_weight_cache.insert(weight_ptr, fp16_buf);
        Ok(fp16_ptr)
    }

    /// PMAT-031: cuBLAS HGEMM prefill — cached FP16 weights + tensor cores
    ///
    /// C[M×N] = Input_fp16[M×K] @ W_fp16[N×K]^T → C is FP32
    ///
    /// Uses gemm_f16_to_f32: FP16 inputs, FP32 output, FP32 accumulation, tensor cores.
    #[allow(clippy::too_many_arguments)]
    fn cublas_prefill_hgemm(
        &mut self,
        w_fp16_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let detail_trace = std::env::var("PREFILL_DETAIL_TRACE").is_ok();
        let t0 = if detail_trace {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Convert FP32 activations → FP16
        let input_count = m as usize * k as usize;
        self.ensure_fp16_activation_scratch(input_count)?;
        let input_fp16_ptr = self
            .fp16_activation_scratch
            .as_ref()
            .expect("scratch just allocated")
            .as_ptr();
        self.convert_f32_to_f16(packed_input_ptr, input_fp16_ptr, input_count as u32)?;

        let t1 = if detail_trace {
            self.stream.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // HGEMM: FP16 weights × FP16 activations → FP32 output (tensor cores)
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
        let result = handle.gemm_f16_to_f32(
            trueno_gpu::driver::GemmOp::Trans,
            trueno_gpu::driver::GemmOp::NoTrans,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            w_fp16_ptr,
            k as i32,
            input_fp16_ptr,
            k as i32,
            0.0,
            packed_output_ptr,
            n as i32,
        );

        if let (Some(t0), Some(t1)) = (t0, t1) {
            self.stream.synchronize()?;
            let t2 = std::time::Instant::now();
            eprintln!(
                "[HGEMM-TRACE] M={} N={} K={}: cvt={:.3}ms cublas={:.3}ms total={:.3}ms",
                m,
                n,
                k,
                t1.duration_since(t0).as_secs_f64() * 1000.0,
                t2.duration_since(t1).as_secs_f64() * 1000.0,
                t2.duration_since(t0).as_secs_f64() * 1000.0,
            );
        }

        result
    }

    /// PMAT-024/026/031/053/064/GH-182: cuBLAS GEMM (or fused Q4K GEMM) for prefill
    ///
    /// C[M×N] = Input[M×K] @ W[N×K]^T
    ///
    /// Priority:
    /// 0. Q4K_WMMA_PREFILL=1 + Q4K → WMMA tensor core Q4K GEMM (3.56x BW savings + tensor cores)
    /// 1. FUSED_Q4K_PREFILL=1 + Q4K → tiled fused Q4K GEMM (reads Q4K directly, scalar FMA)
    /// 2. FP8_PREFILL=1 + sm_89+ → cached FP8 E4M3 weights + cuBLAS FP8 GEMM (1 B/elem, 2x vs HGEMM)
    /// 3. L2_PREFILL=1 + Q4K → per-matmul Q4K→FP16 dequant + L2-cached HGEMM (3.56x less DRAM BW)
    /// 4. HGEMM_PREFILL!=0 (default) → cached FP16 weights + cuBLAS HGEMM + tensor cores
    /// 5. HGEMM_PREFILL=0 → per-request dequant + cuBLAS SGEMM
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cublas_prefill_gemm(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // PMAT-066: DP4A Q4K×Q8 GEMM — no FP16 dequant, 3.56x BW reduction
        if qtype == WeightQuantType::Q4K && std::env::var("DP4A_GEMM_PREFILL").as_deref() == Ok("1")
        {
            return self.launch_dp4a_q4k_gemm(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // PMAT-045: Multi-warp Q4K WMMA GEMM — 4 warps, 32×32 tiles, maxnreg(96)
        if qtype == WeightQuantType::Q4K && std::env::var("MW_WMMA_PREFILL").as_deref() == Ok("1") {
            return self.launch_mw_q4k_wmma_kernel(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // PMAT-064: Q4K WMMA GEMM — tensor cores with direct Q4K weight reads
        // Dequant Q4K→FP16 in SHMEM, WMMA 16×16×16 matmul. 3.56x less BW than HGEMM.
        if qtype == WeightQuantType::Q4K && std::env::var("Q4K_WMMA_PREFILL").as_deref() == Ok("1")
        {
            return self.q4k_wmma_gemm_prefill(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // GH-182: Fused Q4K GEMM — reads Q4K directly (0.5625 B/elem vs 2 B/elem HGEMM)
        if qtype == WeightQuantType::Q4K && std::env::var("FUSED_Q4K_PREFILL").as_deref() == Ok("1")
        {
            return self.fused_q4k_gemm_prefill(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        self.ensure_cublas()?;

        // PMAT-053/067: FP8 E4M3 GEMM — 1 byte/elem (2x BW savings vs HGEMM)
        // Auto-enabled on sm_89+ (Ada Lovelace/Hopper). Override: FP8_PREFILL=0 to disable.
        // PMAT-410: Removed cc < 100 guard — FP8 GEMM works on sm_121 (Blackwell).
        // Warmup skipped on cc >= 100 (attention.rs), but lazy cache in get_or_cache works.
        if self.gpu_profile.fp8_prefill {
            let w_fp8_ptr = self.get_or_cache_fp8_weight(qtype, weight_ptr, n, k)?;
            return self.cublas_prefill_fp8_gemm(
                w_fp8_ptr,
                weight_ptr, // key into fp8_weight_scales
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // PMAT-065: L2-cached HGEMM — per-matmul Q4K→FP16 dequant + HGEMM from L2
        // Reads Q4K from DRAM (0.5625 B/elem), writes FP16 to temp buffer (L2-hot),
        // cuBLAS reads FP16 from L2 instead of DRAM. 3.56x less DRAM bandwidth.
        // Enable with L2_PREFILL=1. Eliminates need for 2944 MB FP16 weight cache.
        if qtype == WeightQuantType::Q4K && std::env::var("L2_PREFILL").as_deref() == Ok("1") {
            let w_fp16_ptr = self.dequant_q4k_fp16_temp(weight_ptr, n, k)?;
            return self.cublas_prefill_hgemm(
                w_fp16_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // PMAT-031: HGEMM path with cached FP16 weights (default)
        // GH-141: Skip HGEMM when FP16 cache was cleared (batched mode frees it
        // to make room for batched KV caches on 8GB GPUs). Uses SGEMM instead.
        if std::env::var("HGEMM_PREFILL").as_deref() != Ok("0")
            && !self.fp16_weight_cache.is_empty()
        {
            let w_fp16_ptr = self.get_or_cache_fp16_weight(qtype, weight_ptr, n, k)?;
            return self.cublas_prefill_hgemm(
                w_fp16_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n,
                k,
            );
        }

        // Fallback: dequant + SGEMM (original PMAT-024/026 path)
        let w_ptr = match qtype {
            WeightQuantType::Q4K => self.dequant_q4k_to_scratch(weight_ptr, n, k)?,
            WeightQuantType::Q6K => self.dequant_q6k_to_scratch(weight_ptr, n, k)?,
            _ => {
                return Err(GpuError::InvalidParameter(format!(
                    "cublas_prefill_gemm: unsupported qtype {:?}",
                    qtype
                )))
            },
        };

        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
        handle.gemm_f32(
            trueno_gpu::driver::GemmOp::Trans,
            trueno_gpu::driver::GemmOp::NoTrans,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            w_ptr,
            k as i32,
            packed_input_ptr,
            k as i32,
            0.0,
            packed_output_ptr,
            n as i32,
        )
    }

    /// PMAT-064: Q4K WMMA GEMM for prefill — tensor cores + direct Q4K reads
    ///
    /// C[M×N] = A[M×K] @ B_q4k[N×(K/256)×144B]^T
    ///
    /// Dequantizes Q4K super-blocks to FP16 in shared memory, uses WMMA
    /// 16×16×16 tensor core tiles for compute. 3.56× less bandwidth than
    /// HGEMM (0.5625 B/elem vs 2 B/elem for FP16).
    ///
    /// Grid: (ceil(N/16), ceil(M/16)), Block: 32 threads (1 warp per WMMA tile)
    #[allow(clippy::too_many_arguments)]
    fn q4k_wmma_gemm_prefill(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        self.launch_q4k_wmma_kernel(weight_ptr, packed_input_ptr, packed_output_ptr, m, n, k)
    }

    /// Launch the Q4K WMMA GEMM kernel
    ///
    /// WMMA stores full 16×16 tiles, so when M or N isn't a multiple of 16,
    /// edge tiles write past the output buffer. To avoid corrupting adjacent
    /// GPU memory, we allocate a padded temporary buffer and copy back.
    #[allow(clippy::too_many_arguments)]
    fn launch_q4k_wmma_kernel(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Pad M and N to multiples of 16 for WMMA tile safety
        let m_padded = (m + 15) & !15;
        let n_padded = (n + 15) & !15;
        let needs_padding = m_padded != m || n_padded != n;

        // Use padded dimensions for kernel compilation (n_const in store stride)
        let kernel_type = KernelType::TensorCoreQ4KGemm {
            m: m_padded,
            n: n_padded,
            k,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("tensor_core_q4k_gemm_{m_padded}_{n_padded}_{k}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        // If padding needed, allocate temp buffer BEFORE borrowing modules
        let actual_output_ptr = if needs_padding {
            let padded_count = m_padded as usize * n_padded as usize;
            self.ensure_wmma_scratch(padded_count)?;
            self.wmma_scratch
                .as_ref()
                .expect("wmma scratch allocated")
                .as_ptr()
        } else {
            packed_output_ptr
        };

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: (ceil(N/16), ceil(M/16)), Block: 32 (1 warp for WMMA)
        let grid_x = n_padded / 16;
        let grid_y = m_padded / 16;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, 32, 1);

        let mut ptr_a = packed_input_ptr;
        let mut ptr_b = weight_ptr;
        let mut ptr_c = actual_output_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

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

        // Copy valid [M, N] from padded buffer to actual output
        if needs_padding {
            // Synchronize stream to ensure WMMA kernel completes before D2D copy.
            // cuMemcpyDtoD is host-synchronous but NOT stream-ordered — it races
            // with async kernel launches without this sync.
            self.stream.synchronize()?;
            // Copy row by row: each row has N valid elements out of N_padded
            for row in 0..m {
                let src_offset = row as u64 * n_padded as u64 * 4;
                let dst_offset = row as u64 * n as u64 * 4;
                self.stream.memcpy_dtod_sync(
                    packed_output_ptr + dst_offset,
                    actual_output_ptr + src_offset,
                    n as usize * 4,
                )?;
            }
        }

        Ok(())
    }

    /// PMAT-045: Multi-Warp Q4K WMMA GEMM — 4 warps, 32×32 output tiles
    ///
    /// C[M×N] = A[M×K] @ B_q4k[N×(K/256)×144B]^T
    ///
    /// 4 warps per block (128 threads), each warp handles a 16×16 WMMA tile.
    /// Grid: (ceil(N/32), ceil(M/32)). SHMEM: 2048 bytes.
    /// maxnreg(96) limits register pressure for better occupancy.
    #[allow(clippy::too_many_arguments)]
    fn launch_mw_q4k_wmma_kernel(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Pad M and N to multiples of 32 for 2×2 WMMA tile safety
        let m_padded = (m + 31) & !31;
        let n_padded = (n + 31) & !31;
        let needs_padding = m_padded != m || n_padded != n;

        let kernel_type = KernelType::MultiWarpTensorCoreQ4KGemm {
            m: m_padded,
            n: n_padded,
            k,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("mw_tensor_core_q4k_gemm_{m_padded}_{n_padded}_{k}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        // If padding needed, allocate temp buffer
        let actual_output_ptr = if needs_padding {
            let padded_count = m_padded as usize * n_padded as usize;
            self.ensure_wmma_scratch(padded_count)?;
            self.wmma_scratch
                .as_ref()
                .expect("wmma scratch allocated")
                .as_ptr()
        } else {
            packed_output_ptr
        };

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: (ceil(N/32), ceil(M/32)), Block: 128 (4 warps for 2×2 WMMA tiles)
        let grid_x = n_padded / 32;
        let grid_y = m_padded / 32;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, 128, 1);

        let mut ptr_a = packed_input_ptr;
        let mut ptr_b = weight_ptr;
        let mut ptr_c = actual_output_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

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

        // Copy valid [M, N] from padded buffer to actual output
        if needs_padding {
            self.stream.synchronize()?;
            for row in 0..m {
                let src_offset = row as u64 * n_padded as u64 * 4;
                let dst_offset = row as u64 * n as u64 * 4;
                self.stream.memcpy_dtod_sync(
                    packed_output_ptr + dst_offset,
                    actual_output_ptr + src_offset,
                    n as usize * 4,
                )?;
            }
        }

        Ok(())
    }

    /// PMAT-066: DP4A Q4K×Q8 GEMM — dequant-free prefill
    ///
    /// Pipeline:
    /// 1. Q8 quantize: f32 activations → Q8_1 format (36 bytes per 32 values)
    /// 2. DP4A GEMM: Q4K weights × Q8 activations → f32 output
    ///
    /// No FP16 dequantization. 3.56x memory bandwidth reduction vs HGEMM.
    #[allow(clippy::too_many_arguments)]
    fn launch_dp4a_q4k_gemm(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let total_f32_elements = m * k;
        let num_q8_blocks = total_f32_elements / 32;
        let q8_bytes = num_q8_blocks as usize * 36;

        // Ensure Q8 scratch buffer is large enough
        let need_alloc = match &self.dp4a_q8_scratch {
            Some(buf) => buf.len() < q8_bytes,
            None => true,
        };
        if need_alloc {
            self.dp4a_q8_scratch = Some(GpuBuffer::<u8>::new(&self.context, q8_bytes)?);
        }
        let q8_ptr = self
            .dp4a_q8_scratch
            .as_ref()
            .expect("q8 scratch allocated")
            .as_ptr();

        // Step 1: Q8 quantize M*K f32 activations → Q8_1
        {
            let kernel_type = KernelType::Q8Quantize {
                n: total_f32_elements,
            };
            let kernel_name = self.kernels.kernel_name(&kernel_type);
            let cache_key = format!("q8_quantize_{total_f32_elements}");

            if !self.modules.contains_key(&cache_key) {
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(cache_key.clone(), module);
            }

            let module = self
                .modules
                .get_mut(&cache_key)
                .expect("module just inserted");
            let config = LaunchConfig::grid_2d(num_q8_blocks, 1, 32, 1);
            let mut out = q8_ptr;
            let mut inp = packed_input_ptr;
            let mut n_val = total_f32_elements;

            unsafe {
                self.stream.launch_kernel(
                    module,
                    kernel_name,
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut out) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut inp) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Step 2: DP4A Q4K×Q8 GEMM
        let num_warps: u32 = 4;
        let num_half_warps = num_warps * 2;
        let tile_m: u32 = 4;

        let kernel_type = KernelType::Dp4aQ4KGemm { m, n, k };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("dp4a_q4k_gemm_{m}_{n}_{k}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let grid_x = (n + num_half_warps - 1) / num_half_warps;
        let grid_y = (m + tile_m - 1) / tile_m;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, num_warps * 32, 1);

        let mut ptr_y = packed_output_ptr;
        let mut ptr_w = weight_ptr;
        let mut ptr_q8 = q8_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_y) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_w) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_q8) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// Ensure WMMA scratch buffer is large enough
    pub(crate) fn ensure_wmma_scratch(&mut self, count: usize) -> Result<(), GpuError> {
        let need_alloc = match &self.wmma_scratch {
            Some(buf) => buf.len() < count,
            None => true,
        };
        if need_alloc {
            self.wmma_scratch = Some(GpuBuffer::<f32>::new(&self.context, count)?);
        }
        Ok(())
    }

    /// GH-182: Fused tiled Q4K GEMM for prefill — reads Q4K weights directly
    ///
    /// C[M×N] = A[M×K] @ B_q4k[N×(K/256)×144B]^T
    ///
    /// Each thread computes tile_m output rows for one column, loading weight
    /// super-blocks once and reusing across rows. 3.56x bandwidth reduction
    /// vs HGEMM (0.5625 B/elem vs 2 B/elem).
    #[allow(clippy::too_many_arguments)]
    fn fused_q4k_gemm_prefill(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let tile_m: u32 = std::env::var("FUSED_Q4K_TILE_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);

        let kernel_type = KernelType::QuantizedGemmGgmlTiled { m, n, k, tile_m };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("q4k_gemm_ggml_tiled_{m}_{n}_{k}_{tile_m}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: (ceil(N/block_threads), ceil(M/tile_m))
        let block_threads = 128u32;
        let grid_x = (n + block_threads - 1) / block_threads;
        let grid_y = (m + tile_m - 1) / tile_m;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, block_threads, 1);

        let mut ptr_a = packed_input_ptr;
        let mut ptr_b = weight_ptr;
        let mut ptr_c = packed_output_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

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

        Ok(())
    }
}
