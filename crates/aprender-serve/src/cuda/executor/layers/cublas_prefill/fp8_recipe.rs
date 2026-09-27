//! #3807: the FP8 prefill recipe — one E4M3 scale per row, subnormals kept.
//!
//! The FP8 GEMM used to quantize the whole [m × k] activation batch with ONE scale,
//! `448/absmax`, and its converters sent every E4M3 subnormal to zero. The down projection's
//! input carries massive-activation outliers (qwen2.5-coder-7b: absmax 871 at layer 27), so
//! everything below `absmax × 3.5e-5` became 0 — up to 12% of a layer's inputs, on every
//! token. `apr run --chat "What is the capital of France? Answer briefly."` on that model then
//! failed the F2 gate at position 1 (cosine 0.8134) and ran on CPU.
//!
//! Now each activation row (token) and each weight row (output channel) gets its own scale,
//! and the quantizer encodes subnormals. Both scales are outer factors of the GEMM, so they are
//! applied after it, in the BF16→F32 widening: `Y[t, c] = D[t, c] × (a[t]/448) × (w[c]/448)`,
//! with the GEMM's alpha at 1. `D` stays BF16 (#3728): it is now at most `448² × k`, well inside
//! BF16's FP32 exponent range. No cuBLASLt scaling mode is needed, and nothing costs more VRAM
//! than a float per row.
//!
//! All three kernels target sm_75 like the rest of this module; FP8 GEMMs only run on sm_89+.

use super::super::super::*;

/// Row absmax of a row-major `[rows × cols]` f32 matrix: one 256-thread block per row, a
/// strided scan, then a shared-memory tree. `out[row] = max |x|` (NaN is ignored by `max.f32`).
const FP8_ABSMAX_ROWS_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry fp8_absmax_rows(
    .param .u64 param_out,
    .param .u64 param_in,
    .param .u32 param_cols
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<10>;
    .reg .f32 %f<3>;
    .reg .pred %p<2>;
    .shared .align 4 .b32 sdata[256];

    ld.param.u64 %rd0, [param_out];
    ld.param.u64 %rd1, [param_in];
    ld.param.u32 %r0, [param_cols];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;

    // this row's base address: in + row * cols * 4
    mul.wide.u32 %rd2, %r2, %r0;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd2, %rd1, %rd2;

    mov.f32 %f0, 0f00000000;
    mov.u32 %r4, %r1;
L_SCAN:
    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra L_REDUCE;
    mul.wide.u32 %rd3, %r4, 4;
    add.u64 %rd3, %rd2, %rd3;
    ld.global.f32 %f1, [%rd3];
    abs.f32 %f1, %f1;
    max.f32 %f0, %f0, %f1;
    add.u32 %r4, %r4, %r3;
    bra L_SCAN;

L_REDUCE:
    mov.u32 %r6, sdata;
    shl.b32 %r9, %r1, 2;
    add.u32 %r7, %r6, %r9;
    st.shared.f32 [%r7], %f0;
    bar.sync 0;
    mov.u32 %r8, 128;
L_TREE:
    setp.ge.u32 %p1, %r1, %r8;
    @%p1 bra L_TREE_SKIP;
    add.u32 %r9, %r1, %r8;
    shl.b32 %r9, %r9, 2;
    add.u32 %r9, %r6, %r9;
    ld.shared.f32 %f2, [%r9];
    max.f32 %f0, %f0, %f2;
    st.shared.f32 [%r7], %f0;
L_TREE_SKIP:
    bar.sync 0;
    shr.u32 %r8, %r8, 1;
    setp.ne.u32 %p1, %r8, 0;
    @%p1 bra L_TREE;

    setp.ne.u32 %p0, %r1, 0;
    @%p0 bra L_EXIT;
    mul.wide.u32 %rd4, %r2, 4;
    add.u64 %rd4, %rd0, %rd4;
    st.global.f32 [%rd4], %f0;
L_EXIT:
    ret;
}
"#;

/// f32 → E4M3 with a per-row scale `448/absmax[idx / cols]` (absmax 0 → scale 448).
/// Round to nearest even, saturating at 448 (`0x7E`; E4M3FN has no infinity and `0x7F` is NaN).
/// Below 2^-6 the value is encoded as a subnormal `q × 2^-9`, `q = rne(|x| × 512)` in 0..=8,
/// and `q = 8` is exactly the smallest normal's code `0x08`. The old converters sent that whole
/// range to zero.
const FP8_QUANTIZE_ROWS_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry fp8_quantize_rows(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count,
    .param .u32 param_cols,
    .param .u64 param_absmax
) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<16>;
    .reg .f32 %f<6>;
    .reg .pred %p<4>;

    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    ld.param.u32 %r15, [param_cols];
    ld.param.u64 %rd4, [param_absmax];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE;

    // scale = 448 / absmax[row]
    div.u32 %r2, %r1, %r15;
    mul.wide.u32 %rd5, %r2, 4;
    add.u64 %rd5, %rd4, %rd5;
    ld.global.f32 %f4, [%rd5];
    setp.eq.f32 %p3, %f4, 0f00000000;
    @%p3 mov.f32 %f4, 0f3F800000;
    mov.f32 %f5, 0f43E00000;
    div.rn.f32 %f3, %f5, %f4;

    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];
    mul.f32 %f0, %f0, %f3;

    // sign bit, then |x| saturated at 448 (min.f32 also sends NaN to 448)
    mov.b32 %r4, %f0;
    shr.u32 %r5, %r4, 31;
    shl.b32 %r5, %r5, 7;
    abs.f32 %f1, %f0;
    min.f32 %f1, %f1, 0f43E00000;
    setp.lt.f32 %p1, %f1, 0f3C800000;
    @%p1 bra L_SUBNORMAL;

    // normal: biased exponent 1..15, 3 mantissa bits, round to nearest even with carry
    mov.b32 %r4, %f1;
    bfe.u32 %r6, %r4, 23, 8;
    sub.u32 %r8, %r6, 120;
    and.b32 %r7, %r4, 0x007FFFFF;
    shr.u32 %r9, %r7, 20;
    bfe.u32 %r10, %r7, 19, 1;
    and.b32 %r11, %r7, 0x0007FFFF;
    and.b32 %r12, %r9, 1;
    or.b32 %r13, %r11, %r12;
    setp.ne.u32 %p2, %r13, 0;
    selp.u32 %r14, %r10, 0, %p2;
    add.u32 %r9, %r9, %r14;
    setp.gt.u32 %p2, %r9, 7;
    @!%p2 bra L_PACK;
    mov.u32 %r9, 0;
    add.u32 %r8, %r8, 1;
L_PACK:
    shl.b32 %r8, %r8, 3;
    or.b32 %r5, %r5, %r8;
    or.b32 %r5, %r5, %r9;
    bra L_STORE;

L_SUBNORMAL:
    mul.f32 %f2, %f1, 0f44000000;
    cvt.rni.u32.f32 %r9, %f2;
    or.b32 %r5, %r5, %r9;

L_STORE:
    add.u64 %rd4, %rd0, %rd2;
    st.global.u8 [%rd4], %r5;
L_DONE:
    ret;
}
"#;

/// The GEMM's BF16 `D` (token-major: element `(c, t)` at `t × n + c`) widened to f32 and
/// multiplied by both outer scales: `Y = D × (act_absmax[t]/448) × (w_absmax[c]/448)`.
/// BF16 is the top half of an f32, so the widening is an exact 16-bit shift (sm_75, #3728).
const FP8_DEQUANT_OUTER_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry fp8_dequant_outer(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count,
    .param .u32 param_n,
    .param .u64 param_act_absmax,
    .param .u64 param_w_absmax
) {
    .reg .u64 %rd<8>;
    .reg .u32 %r<8>;
    .reg .u16 %hs;
    .reg .f32 %f<4>;
    .reg .pred %p<3>;

    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    ld.param.u32 %r6, [param_n];
    ld.param.u64 %rd5, [param_act_absmax];
    ld.param.u64 %rd6, [param_w_absmax];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE;

    // token t = idx / n, channel c = idx - t * n
    div.u32 %r4, %r1, %r6;
    mul.lo.u32 %r5, %r4, %r6;
    sub.u32 %r5, %r1, %r5;
    mul.wide.u32 %rd7, %r4, 4;
    add.u64 %rd7, %rd5, %rd7;
    ld.global.f32 %f1, [%rd7];
    mul.wide.u32 %rd7, %r5, 4;
    add.u64 %rd7, %rd6, %rd7;
    ld.global.f32 %f2, [%rd7];
    // an all-zero row was quantized with absmax 1 (see fp8_quantize_rows); keep them paired
    setp.eq.f32 %p1, %f1, 0f00000000;
    @%p1 mov.f32 %f1, 0f3F800000;
    setp.eq.f32 %p2, %f2, 0f00000000;
    @%p2 mov.f32 %f2, 0f3F800000;
    mov.f32 %f3, 0f43E00000;
    div.rn.f32 %f1, %f1, %f3;
    div.rn.f32 %f2, %f2, %f3;

    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 1;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.u16 %hs, [%rd3];
    cvt.u32.u16 %r7, %hs;
    shl.b32 %r7, %r7, 16;
    mov.b32 %f0, %r7;
    mul.f32 %f0, %f0, %f1;
    mul.f32 %f0, %f0, %f2;
    shl.b64 %rd4, %rd2, 2;
    add.u64 %rd4, %rd0, %rd4;
    st.global.f32 [%rd4], %f0;
L_DONE:
    ret;
}
"#;

impl CudaExecutor {
    /// Compile one of this recipe's constant-PTX modules once, under its own name.
    fn ensure_fp8_recipe_module(&mut self, key: &str, ptx: &str) -> Result<(), GpuError> {
        if !self.modules.contains_key(key) {
            let module = self.compile_ptx(ptx)?;
            self.modules.insert(key.to_string(), module);
        }
        Ok(())
    }

    /// `out[row] = max |src[row, ..]|` for a row-major `[rows × cols]` f32 matrix on the device.
    pub(super) fn fp8_absmax_rows(
        &mut self,
        src_ptr: u64,
        rows: u32,
        cols: u32,
        out_ptr: u64,
    ) -> Result<(), GpuError> {
        self.ensure_fp8_recipe_module("fp8_absmax_rows", FP8_ABSMAX_ROWS_PTX)?;
        let module = self
            .modules
            .get_mut("fp8_absmax_rows")
            .expect("compiled above");
        // The kernel's tree assumes exactly 256 threads per block.
        let config = LaunchConfig {
            grid: (rows, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        };
        let mut out = out_ptr;
        let mut src = src_ptr;
        let mut cols = cols;
        // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature; `src_ptr` holds `rows × cols` f32 and `out_ptr` holds `rows` f32, both allocated by the caller and alive until the stream-ordered launch completes.
        unsafe {
            self.stream.launch_kernel(
                module,
                "fp8_absmax_rows",
                &config,
                &mut [
                    std::ptr::from_mut(&mut out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cols) as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(())
    }

    /// E4M3-encode a row-major `[rows × cols]` f32 matrix, each row with scale `448/absmax[row]`.
    pub(super) fn fp8_quantize_rows(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        rows: u32,
        cols: u32,
        absmax_ptr: u64,
    ) -> Result<(), GpuError> {
        self.ensure_fp8_recipe_module("fp8_quantize_rows", FP8_QUANTIZE_ROWS_PTX)?;
        let module = self
            .modules
            .get_mut("fp8_quantize_rows")
            .expect("compiled above");
        let count = rows * cols;
        let config = LaunchConfig::linear(count, 256);
        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;
        let mut cols = cols;
        let mut absmax = absmax_ptr;
        // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature; `src_ptr` holds `rows × cols` f32, `dst_ptr` at least `rows × cols` bytes and `absmax_ptr` `rows` f32, all alive until the stream-ordered launch completes.
        unsafe {
            self.stream.launch_kernel(
                module,
                "fp8_quantize_rows",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cols) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut absmax) as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(())
    }

    /// Widen the FP8 GEMM's BF16 output for `m` tokens × `n` channels to f32 and apply both
    /// outer scales (`act_absmax[m]`, `w_absmax[n]`, each over 448).
    pub(super) fn fp8_dequant_outer(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        m: u32,
        n: u32,
        act_absmax_ptr: u64,
        w_absmax_ptr: u64,
    ) -> Result<(), GpuError> {
        self.ensure_fp8_recipe_module("fp8_dequant_outer", FP8_DEQUANT_OUTER_PTX)?;
        let module = self
            .modules
            .get_mut("fp8_dequant_outer")
            .expect("compiled above");
        let count = m * n;
        let config = LaunchConfig::linear(count, 256);
        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;
        let mut n = n;
        let mut act = act_absmax_ptr;
        let mut w = w_absmax_ptr;
        // SAFETY: launches a CUDA kernel via the driver API. The argument pointer array, grid/block config, and module/function name match the kernel's signature; `src_ptr` holds at least `m × n` bf16, `dst_ptr` `m × n` f32, `act_absmax_ptr` `m` f32 and `w_absmax_ptr` `n` f32, all alive until the stream-ordered launch completes.
        unsafe {
            self.stream.launch_kernel(
                module,
                "fp8_dequant_outer",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut act) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut w) as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(())
    }

    /// Quantize a row-major `[n × k]` f32 weight on the device to E4M3 with one scale per output
    /// channel. Returns the FP8 weight and its per-channel absmax, which the GEMM's dequant needs.
    pub(super) fn fp8_quantize_weight(
        &mut self,
        f32_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<(GpuBuffer<u8>, GpuBuffer<f32>), GpuError> {
        let row_absmax = GpuBuffer::<f32>::new(&self.context, n as usize)?;
        self.fp8_absmax_rows(f32_ptr, n, k, row_absmax.as_ptr())?;
        let fp8 = GpuBuffer::<u8>::new(&self.context, n as usize * k as usize)?;
        self.fp8_quantize_rows(f32_ptr, fp8.as_ptr(), n, k, row_absmax.as_ptr())?;
        Ok((fp8, row_absmax))
    }

    /// The per-token activation absmax buffer, grown to at least `rows` floats. Growing it drops
    /// the PMAT-084 held activation, whose row scales lived in the old buffer.
    pub(super) fn ensure_fp8_act_row_absmax(&mut self, rows: usize) -> Result<u64, GpuError> {
        let big_enough = self
            .fp8_act_row_absmax
            .as_ref()
            .is_some_and(|b| b.len() >= rows);
        if !big_enough {
            self.fp8_act_row_absmax = Some(GpuBuffer::<f32>::new(&self.context, rows.max(16))?);
            self.fp8_act_cache.invalidate();
        }
        Ok(self
            .fp8_act_row_absmax
            .as_ref()
            .expect("allocated above")
            .as_ptr())
    }
}
