//! PMAT-024/026/031: cuBLAS GEMM for prefill
//!
//! Prefill path evolution:
//!   v1 (PMAT-024): Dequant Q4K/Q6K → FP32 scratch → cuBLAS SGEMM (per-request dequant)
//!   v2 (PMAT-031): Cache FP16 weights + cuBLAS HGEMM with tensor cores
//!
//! PMAT-031 eliminates per-request dequant cost and uses tensor cores for 2-4x speedup.
//! On first prefill, weights are dequantized to FP32, converted to FP16, and cached.
//! Subsequent prefills use cached FP16 weights directly with HGEMM.
//!
//! Split from monolithic file for maintainability (Refs PMAT-149):
//!   mod.rs — PTX constants, threshold, setup/conversion helpers
//!   gemm.rs — GEMM dispatch (FP8, HGEMM, WMMA, fused Q4K)
//!   attention.rs — prefill attention, KV scatter, cache management, warmup

mod attention;
mod fp8_recipe;
mod gemm;
// #3728: the FP8 GEMM's intermediate range, on the real kernel.
#[cfg(test)]
mod fp8_gemm_range_tests_3728;
// #3807: per-row E4M3 scales and subnormals, on the real kernels.
#[cfg(test)]
mod fp8_recipe_tests_3807;

use super::super::*;

/// Minimum M (batch/sequence length) to use cuBLAS GEMM instead of batched GEMV.
/// Below this threshold, batched GEMV is faster due to lower overhead.
///
/// Override with CUBLAS_GEMM_THRESHOLD env var (e.g., =1 for HGEMM decode on high-BW GPUs).
/// Minimum batch size M for cuBLAS SGEMM to beat batched GEMV in decode.
///
/// GH-141 Five-Whys: At M=4, cuBLAS SGEMM dequants Q4K → FP32 (4 B/elem)
/// then runs SGEMM. Batched GEMV reads Q4K directly (0.5625 B/elem) — 7.1x
/// less bandwidth. SGEMM only wins at large M where compute dominates.
///
/// Default=4: cuBLAS SGEMM beats batched GEMV at M=4 (51 vs 35 tok/s).
/// Batched Q4K GEMV at M<=8 uses single warp (32 threads/block) — insufficient
/// parallelism. Multi-warp specializations only exist for M=16/32.
/// Deferred (PMAT-759): Add M=4 multi-warp kernel, then raise threshold to 8+.
pub(crate) fn cublas_gemm_threshold() -> u32 {
    std::env::var("CUBLAS_GEMM_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

/// PMAT-031: Inline PTX for FP32→FP16 element-wise conversion.
/// Block size 256, one element per thread. Trivially memory-bound (~1μs for 160K elements).
const F32_TO_F16_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f32_to_f16(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<4>;
    .reg .f32 %f0;
    .reg .b16 %h0;
    .reg .pred %p0;
    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE;
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];
    cvt.rn.f16.f32 %h0, %f0;
    shl.b64 %rd4, %rd2, 1;
    add.u64 %rd4, %rd0, %rd4;
    st.global.b16 [%rd4], %h0;
L_DONE:
    ret;
}
"#;

/// PMAT-053: Inline PTX for FP16->FP32 element-wise conversion.
/// Block size 256, one element per thread. Uses hardware cvt.f32.f16.
const F16_TO_F32_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f16_to_f32(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<4>;
    .reg .f32 %f0;
    .reg .b16 %h0;
    .reg .pred %p0;
    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE;
    // Load FP16 (16 bits)
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 1;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.b16 %h0, [%rd3];
    // Convert FP16 -> FP32 (hardware instruction)
    cvt.f32.f16 %f0, %h0;
    // Store as FP32 (4 bytes)
    shl.b64 %rd4, %rd2, 2;
    add.u64 %rd4, %rd0, %rd4;
    st.global.f32 [%rd4], %f0;
L_DONE:
    ret;
}
"#;

impl CudaExecutor {
    /// Initialize cuBLAS handle for prefill GEMM (lazy, called once)
    pub(crate) fn ensure_cublas(&mut self) -> Result<(), GpuError> {
        if self.cublas_handle.is_some() {
            return Ok(());
        }
        let handle = trueno_gpu::driver::CublasHandle::new(&self.context)?;
        handle.set_stream(&self.stream)?;
        self.cublas_handle = Some(handle);
        Ok(())
    }

    /// PMAT-063: Pre-allocate cuBLAS workspace for CUDA graph capture.
    ///
    /// cuBLAS internally allocates workspace for fast algorithm selection.
    /// During CUDA graph capture, dynamic allocation is forbidden, so cuBLAS
    /// falls back to workspace-free algorithms (7x slower on RTX 4060L).
    ///
    /// This allocates a 32 MB GPU buffer and registers it with cuBLAS via
    /// `cublasSetWorkspace`. Must be called before prefill graph capture.
    pub(crate) fn ensure_cublas_workspace(&mut self) -> Result<(), GpuError> {
        if self.cublas_workspace.is_some() {
            return Ok(());
        }
        self.ensure_cublas()?;

        // 32 MB workspace — covers all standard cuBLAS GEMM shapes
        const WORKSPACE_SIZE: usize = 32 * 1024 * 1024;
        let workspace = GpuBuffer::<u8>::new(&self.context, WORKSPACE_SIZE)?;
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");
        handle.set_workspace(workspace.as_ptr(), WORKSPACE_SIZE)?;
        eprintln!(
            "[PMAT-063] cuBLAS workspace: {} MB pre-allocated for graph capture",
            WORKSPACE_SIZE / 1024 / 1024
        );
        self.cublas_workspace = Some(workspace);
        Ok(())
    }

    /// Ensure dequant scratch buffer is large enough for N×K FP32 elements
    fn ensure_dequant_scratch(&mut self, n: u32, k: u32) -> Result<(), GpuError> {
        let needed = n as usize * k as usize;
        if self.dequant_scratch_size >= needed {
            return Ok(());
        }
        self.dequant_scratch = Some(GpuBuffer::new(&self.context, needed)?);
        self.dequant_scratch_size = needed;
        Ok(())
    }

    /// PMAT-031: Ensure FP16 activation scratch is large enough
    /// GH-174: pub(crate) for gemm_b_cached_f16() access
    pub(crate) fn ensure_fp16_activation_scratch(&mut self, count: usize) -> Result<(), GpuError> {
        if self.fp16_activation_scratch_size >= count {
            return Ok(());
        }
        self.fp16_activation_scratch = Some(GpuBuffer::new(&self.context, count)?);
        self.fp16_activation_scratch_size = count;
        Ok(())
    }

    /// PMAT-031: Convert FP32 GPU data to FP16 using inline PTX kernel
    /// GH-174: pub(crate) for gemm_b_cached_f16() access
    pub(crate) fn convert_f32_to_f16(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
    ) -> Result<(), GpuError> {
        // Compile conversion kernel once
        if !self.modules.contains_key("f32_to_f16") {
            let module = self.compile_ptx(F32_TO_F16_PTX)?;
            self.modules.insert("f32_to_f16".to_string(), module);
        }

        let module = self.modules.get_mut("f32_to_f16").expect("just inserted");
        let config = LaunchConfig::linear(count, 256);

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        // SAFETY: src_ptr and dst_ptr are valid GPU allocations, count verified by caller
        unsafe {
            self.stream.launch_kernel(
                module,
                "f32_to_f16",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-053: Ensure FP8 activation scratch is large enough
    fn ensure_fp8_activation_scratch(&mut self, count: usize) -> Result<(), GpuError> {
        if self.fp8_activation_scratch_size >= count {
            return Ok(());
        }
        self.fp8_activation_scratch = Some(GpuBuffer::new(&self.context, count)?);
        self.fp8_activation_scratch_size = count;
        // PMAT-084: Reallocation invalidates cached FP8 activation data
        self.fp8_act_cache.invalidate();
        Ok(())
    }

    /// PMAT-053: Convert FP16 GPU data to FP32 using hardware cvt.f32.f16
    fn convert_f16_to_f32(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
    ) -> Result<(), GpuError> {
        if !self.modules.contains_key("f16_to_f32") {
            let module = self.compile_ptx(F16_TO_F32_PTX)?;
            self.modules.insert("f16_to_f32".to_string(), module);
        }

        let module = self.modules.get_mut("f16_to_f32").expect("just inserted");
        let config = LaunchConfig::linear(count, 256);

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature, and every referenced device buffer is allocated, correctly sized, and lives until the stream-ordered launch completes.
        unsafe {
            self.stream.launch_kernel(
                module,
                "f16_to_f32",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PERF-050 (aprender#2753): define the part of the FP8 activation scratch that the GEMM
    /// READS but nothing WRITES.
    ///
    /// cuBLASLt FP8 needs the batch dimension aligned to 16, so `cublas_prefill_fp8_gemm` asks
    /// for `m_padded * k` elements of scratch and issues the GEMM with `m_padded` as its batch
    /// dimension -- but the conversion above only writes `m * k`. At m = 1 that is 1 row written
    /// and 15 read. The tail is not inert: E4M3 has NaN encodings, and what is actually sitting
    /// there is the previous GEMM's activations, so the GEMM's input depends on call history.
    ///
    /// That is the mechanism behind the batched forward not being a function of its inputs. The
    /// first call after prefill reads prefill's leftovers; the second reads the first call's.
    ///
    /// Zeroing happens HERE, at the end of the conversion, and not in
    /// `ensure_fp8_activation_scratch`. The scratch allocator runs on EVERY GEMM including the
    /// PMAT-084 cache hits, where K and V deliberately reuse Q's converted activation -- zeroing
    /// there wipes exactly the data the hit exists to reuse. That was tried, and it degraded the
    /// output instead of fixing it. The conversion only runs on a cache MISS, so the prefix a
    /// hit reuses stays intact and the tail behind it stays zero.
    ///
    /// DEFAULT OFF, opt in with `APR_FP8_PAD_ZERO=1`. The padding read is a real defect, but it
    /// is NOT the cause of the non-determinism this was written to chase: measured in ONE binary
    /// via this flag, the CB-006 SELF control fails either way (cosine 0.286372 with the padding
    /// left undefined, 0.563568 with it zeroed -- both far from the 1.0 that identical inputs
    /// require). Defaulting it on would change the numerics of every FP8 GEMM and add a host
    /// copy per conversion, on no evidence of benefit. It needs its own ticket and its own
    /// measurement.
    fn zero_fp8_activation_tail(&mut self, dst_ptr: u64, written: u32) -> Result<(), GpuError> {
        if !Self::fp8_pad_zero_enabled() {
            return Ok(());
        }
        let size = self.fp8_activation_scratch_size;
        let written = written as usize;
        let Some(buf) = self.fp8_activation_scratch.as_mut() else {
            return Ok(());
        };
        // Only this scratch is padded; other conversion targets are exact.
        if buf.as_ptr() != dst_ptr || size <= written {
            return Ok(());
        }
        let zeros = vec![0u8; size - written];
        buf.copy_from_host_at(&zeros, written)?;
        Ok(())
    }

    /// PERF-050: `APR_FP8_PAD_ZERO=1` enables zeroing the FP8 activation padding (default off).
    fn fp8_pad_zero_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("APR_FP8_PAD_ZERO").as_deref() == Ok("1"))
    }
}
