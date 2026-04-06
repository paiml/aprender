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
mod gemm;

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
/// TODO: Add M=4 multi-warp kernel, then raise threshold to 8+.
pub(crate) fn cublas_gemm_threshold() -> u32 {
    std::env::var("CUBLAS_GEMM_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

/// PMAT-053: Inline PTX for FP32→FP8 E4M3 element-wise conversion.
/// Works on sm_75+ via manual bit manipulation (no cvt.e4m3x2 which needs sm_90).
/// Each thread converts 1 FP32 → 1 E4M3 byte. Block size 256.
///
/// FP8 E4M3: sign(1) + exponent(4) + mantissa(3), bias=7, range ±448.
/// FP32:     sign(1) + exponent(8) + mantissa(23), bias=127.
///
/// Algorithm: clamp |x| to [2^-9, 448], rebias exponent (127→7), round mantissa
/// (23→3 bits with RNE), pack sign+exp+mantissa into 8 bits.
/// NaN/Inf → max finite (0x7E = 448.0). Zero → 0x00.
const F32_TO_E4M3_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f32_to_e4m3(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<16>;
    .reg .f32 %f<4>;
    .reg .pred %p<4>;

    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];

    // Global thread index
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;

    // Bounds check
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE;

    // Load FP32
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];

    // Reinterpret as u32
    mov.b32 %r4, %f0;

    // Extract sign bit (bit 31) -> %r5
    shr.u32 %r5, %r4, 31;

    // Extract FP32 exponent (bits 30:23) -> %r6
    bfe.u32 %r6, %r4, 23, 8;

    // Extract FP32 mantissa (bits 22:0) -> %r7
    and.b32 %r7, %r4, 0x007FFFFF;

    // Check for zero/denorm (exp == 0) -> output 0x00
    setp.eq.u32 %p1, %r6, 0;
    @%p1 bra L_ZERO;

    // Check for NaN/Inf (exp == 255) -> output sign | 0x7E (max finite)
    setp.eq.u32 %p2, %r6, 255;
    @%p2 bra L_NANINF;

    // Rebias exponent: e4m3_exp = fp32_exp - 127 + 7 = fp32_exp - 120
    // E4M3 valid biased exp range: 1..15 (unbiased -6..+8)
    sub.u32 %r8, %r6, 120;

    // Check underflow (fp32_exp < 121 -> e4m3_exp < 1 -> denorm/zero)
    setp.lt.s32 %p1, %r8, 1;
    @%p1 bra L_ZERO;

    // Check overflow (e4m3_exp > 15 -> clamp to max finite)
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF;

    // Round mantissa: 23 bits -> 3 bits (drop 20 low bits, RNE)
    // Round bit = bit 19, sticky = bits 18:0
    shr.u32 %r9, %r7, 20;          // top 3 mantissa bits
    bfe.u32 %r10, %r7, 19, 1;      // round bit
    and.b32 %r11, %r7, 0x0007FFFF; // sticky bits (bits 18:0)

    // RNE: round up if (round && (sticky || lsb_of_result))
    and.b32 %r12, %r9, 1;          // lsb of truncated mantissa
    or.b32 %r13, %r11, %r12;       // sticky | lsb
    setp.ne.u32 %p3, %r13, 0;
    and.b32 %r14, %r10, 1;         // round bit
    // Only round up if round bit is set AND (sticky|lsb)
    selp.u32 %r14, %r14, 0, %p3;
    add.u32 %r9, %r9, %r14;

    // Handle mantissa overflow (0b1000 -> increment exponent)
    setp.gt.u32 %p3, %r9, 7;
    @!%p3 bra L_PACK;
    mov.u32 %r9, 0;
    add.u32 %r8, %r8, 1;
    // If exponent overflows to 16 -> max finite
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF;

L_PACK:
    // Pack: sign(1) | exp(4) | mantissa(3)
    shl.b32 %r5, %r5, 7;
    shl.b32 %r8, %r8, 3;
    or.b32 %r5, %r5, %r8;
    or.b32 %r5, %r5, %r9;
    bra L_STORE;

L_ZERO:
    mov.u32 %r5, 0;
    bra L_STORE;

L_NANINF:
    // sign | 0x7E (max finite = 448.0)
    shl.b32 %r5, %r5, 7;
    or.b32 %r5, %r5, 0x7E;

L_STORE:
    // Store 1 byte
    add.u64 %rd4, %rd0, %rd2;
    st.global.u8 [%rd4], %r5;

L_DONE:
    ret;
}
"#;

/// PMAT-053b: Scaled FP32→FP8 E4M3 conversion kernel.
/// Same as F32_TO_E4M3_PTX but multiplies input by a per-tensor scale factor first.
/// quant_scale = 448 / absmax — normalizes input range to E4M3 representable range.
const F32_TO_E4M3_SCALED_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f32_to_e4m3_scaled(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count,
    .param .f32 param_scale
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<16>;
    .reg .f32 %f<4>;
    .reg .pred %p<4>;

    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    ld.param.f32 %f3, [param_scale];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;

    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE_S;

    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];
    mul.f32 %f0, %f0, %f3;

    mov.b32 %r4, %f0;
    shr.u32 %r5, %r4, 31;
    bfe.u32 %r6, %r4, 23, 8;
    and.b32 %r7, %r4, 0x007FFFFF;

    setp.eq.u32 %p1, %r6, 0;
    @%p1 bra L_ZERO_S;
    setp.eq.u32 %p2, %r6, 255;
    @%p2 bra L_NANINF_S;

    sub.u32 %r8, %r6, 120;
    setp.lt.s32 %p1, %r8, 1;
    @%p1 bra L_ZERO_S;
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF_S;

    shr.u32 %r9, %r7, 20;
    bfe.u32 %r10, %r7, 19, 1;
    and.b32 %r11, %r7, 0x0007FFFF;
    and.b32 %r12, %r9, 1;
    or.b32 %r13, %r11, %r12;
    setp.ne.u32 %p3, %r13, 0;
    and.b32 %r14, %r10, 1;
    selp.u32 %r14, %r14, 0, %p3;
    add.u32 %r9, %r9, %r14;

    setp.gt.u32 %p3, %r9, 7;
    @!%p3 bra L_PACK_S;
    mov.u32 %r9, 0;
    add.u32 %r8, %r8, 1;
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF_S;

L_PACK_S:
    shl.b32 %r5, %r5, 7;
    shl.b32 %r8, %r8, 3;
    or.b32 %r5, %r5, %r8;
    or.b32 %r5, %r5, %r9;
    bra L_STORE_S;

L_ZERO_S:
    mov.u32 %r5, 0;
    bra L_STORE_S;

L_NANINF_S:
    shl.b32 %r5, %r5, 7;
    or.b32 %r5, %r5, 0x7E;

L_STORE_S:
    add.u64 %rd4, %rd0, %rd2;
    st.global.u8 [%rd4], %r5;

L_DONE_S:
    ret;
}
"#;

/// PMAT-079: Device-scaled FP32→FP8 E4M3 conversion kernel.
/// Reads absmax from a device pointer (no CPU sync needed).
/// Thread 0 of block 0 also writes dequant_scale = absmax/448 to a device buffer
/// for use as cuBLASLt A_SCALE_POINTER.
const F32_TO_E4M3_DEVICE_SCALED_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f32_to_e4m3_device_scaled(
    .param .u64 param_dst,           // FP8 output buffer
    .param .u64 param_src,           // FP32 input buffer
    .param .u32 param_count,         // number of elements
    .param .u64 param_absmax_ptr,    // device ptr to u32 (absmax as IEEE 754 bits)
    .param .u64 param_dequant_ptr    // device ptr to f32 where to write absmax/448
) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<16>;
    .reg .f32 %f<7>;
    .reg .pred %p<5>;

    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    ld.param.u64 %rd4, [param_absmax_ptr];
    ld.param.u64 %rd5, [param_dequant_ptr];

    // Read absmax from device memory (u32 bits to f32)
    ld.global.u32 %r15, [%rd4];
    mov.b32 %f4, %r15;

    // Handle absmax == 0: use 1.0 to avoid div-by-zero
    setp.eq.f32 %p3, %f4, 0f00000000;
    @%p3 mov.f32 %f4, 0f3F800000;

    // Compute quant_scale = 448 / absmax (f5 = 448.0, reused below)
    mov.f32 %f5, 0f43E00000;
    div.rn.f32 %f3, %f5, %f4;

    // Thread 0 of block 0: write dequant_scale = absmax / 448 to device buffer
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    or.b32 %r14, %r1, %r2;
    setp.ne.u32 %p4, %r14, 0;
    @%p4 bra L_SKIP_DEQUANT;
    div.rn.f32 %f6, %f4, %f5;
    st.global.f32 [%rd5], %f6;
L_SKIP_DEQUANT:

    // Grid-stride loop for FP8 conversion
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;

    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE_DS;

    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f0, [%rd3];
    mul.f32 %f0, %f0, %f3;

    // FP32 to FP8 E4M3 conversion (same logic as f32_to_e4m3_scaled)
    mov.b32 %r4, %f0;
    shr.u32 %r5, %r4, 31;
    bfe.u32 %r6, %r4, 23, 8;
    and.b32 %r7, %r4, 0x007FFFFF;

    setp.eq.u32 %p1, %r6, 0;
    @%p1 bra L_ZERO_DS;
    setp.eq.u32 %p2, %r6, 255;
    @%p2 bra L_NANINF_DS;

    sub.u32 %r8, %r6, 120;
    setp.lt.s32 %p1, %r8, 1;
    @%p1 bra L_ZERO_DS;
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF_DS;

    shr.u32 %r9, %r7, 20;
    bfe.u32 %r10, %r7, 19, 1;
    and.b32 %r11, %r7, 0x0007FFFF;
    and.b32 %r12, %r9, 1;
    or.b32 %r13, %r11, %r12;
    setp.ne.u32 %p3, %r13, 0;
    and.b32 %r14, %r10, 1;
    selp.u32 %r14, %r14, 0, %p3;
    add.u32 %r9, %r9, %r14;

    setp.gt.u32 %p3, %r9, 7;
    @!%p3 bra L_PACK_DS;
    mov.u32 %r9, 0;
    add.u32 %r8, %r8, 1;
    setp.gt.s32 %p2, %r8, 15;
    @%p2 bra L_NANINF_DS;

L_PACK_DS:
    shl.b32 %r5, %r5, 7;
    shl.b32 %r8, %r8, 3;
    or.b32 %r5, %r5, %r8;
    or.b32 %r5, %r5, %r9;
    bra L_STORE_DS;

L_ZERO_DS:
    mov.u32 %r5, 0;
    bra L_STORE_DS;

L_NANINF_DS:
    shl.b32 %r5, %r5, 7;
    or.b32 %r5, %r5, 0x7E;

L_STORE_DS:
    add.u64 %rd4, %rd0, %rd2;
    st.global.u8 [%rd4], %r5;

L_DONE_DS:
    ret;
}
"#;

/// PMAT-053b: GPU absmax reduction kernel.
/// Each block computes a local absmax via shared memory, then atomicMax to global result.
/// Output buffer must be pre-zeroed (single u32 interpreted as FP32 via IEEE 754 ordering).
/// For positive floats, u32 ordering matches f32 ordering, so atom.max.u32 is correct.
const ABSMAX_REDUCE_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry absmax_reduce(
    .param .u64 param_output,
    .param .u64 param_input,
    .param .u32 param_count
) {
    .reg .u64 %rd<5>;
    .reg .u32 %r<10>;
    .reg .f32 %f<4>;
    .reg .pred %p<2>;
    .shared .align 4 .b32 sdata[256];

    ld.param.u64 %rd0, [param_output];
    ld.param.u64 %rd1, [param_input];
    ld.param.u32 %r0, [param_count];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r4, %r2, %r3, %r1;

    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r5, %r5, %r3;

    mov.b32 %f0, 0x00000000;

L_LOOP:
    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra L_REDUCE;

    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd3, %rd2, 2;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f1, [%rd3];
    abs.f32 %f1, %f1;
    max.f32 %f0, %f0, %f1;

    add.u32 %r4, %r4, %r5;
    bra L_LOOP;

L_REDUCE:
    // Store thread-local absmax into shared memory
    // %r7 = &sdata[tid.x] (reused as this thread's shared addr)
    mov.u32 %r6, sdata;
    shl.b32 %r9, %r1, 2;
    add.u32 %r7, %r6, %r9;
    st.shared.f32 [%r7], %f0;
    bar.sync 0;

    // Tree reduction in shared memory
    mov.u32 %r8, 128;
L_RED_LOOP:
    setp.ge.u32 %p1, %r1, %r8;
    @%p1 bra L_RED_DONE;
    add.u32 %r9, %r1, %r8;
    shl.b32 %r9, %r9, 2;
    add.u32 %r9, %r6, %r9;
    ld.shared.f32 %f2, [%r9];
    max.f32 %f0, %f0, %f2;
    st.shared.f32 [%r7], %f0;
L_RED_DONE:
    bar.sync 0;
    shr.u32 %r8, %r8, 1;
    setp.ne.u32 %p1, %r8, 0;
    @%p1 bra L_RED_LOOP;

    // Thread 0 does atomic max into global output
    setp.ne.u32 %p0, %r1, 0;
    @%p0 bra L_EXIT;
    mov.b32 %r9, %f0;
    atom.global.max.u32 %r9, [%rd0], %r9;

L_EXIT:
    ret;
}
"#;

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

/// PMAT-079: FP16 to FP32 conversion with device-side activation dequant scaling.
/// Reads a single FP32 scale from a device pointer (act_dequant = act_absmax/448),
/// multiplies each converted element by it.
/// Weight dequant is already folded into the GEMM alpha — only act_dequant varies per request.
const F16_TO_F32_ACT_SCALED_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry f16_to_f32_act_scaled(
    .param .u64 param_dst,
    .param .u64 param_src,
    .param .u32 param_count,
    .param .u64 param_act_dequant_ptr
) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<2>;
    .reg .b16 %h0;
    .reg .pred %p0;
    ld.param.u64 %rd0, [param_dst];
    ld.param.u64 %rd1, [param_src];
    ld.param.u32 %r0, [param_count];
    ld.param.u64 %rd5, [param_act_dequant_ptr];

    // Read act_dequant scale from device (same for all elements)
    ld.global.f32 %f1, [%rd5];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r1, %r2, %r3, %r1;
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra L_DONE_AS;
    // Load FP16 (16 bits)
    cvt.u64.u32 %rd2, %r1;
    shl.b64 %rd3, %rd2, 1;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.b16 %h0, [%rd3];
    // Convert FP16 to FP32 then scale by act_dequant
    cvt.f32.f16 %f0, %h0;
    mul.f32 %f0, %f0, %f1;
    // Store as FP32 (4 bytes)
    shl.b64 %rd4, %rd2, 2;
    add.u64 %rd4, %rd0, %rd4;
    st.global.f32 [%rd4], %f0;
L_DONE_AS:
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
        self.fp8_activation_cache_key = None;
        Ok(())
    }

    /// PMAT-053: Convert FP32 GPU data to FP8 E4M3 using inline PTX kernel (sm_75+)
    fn convert_f32_to_e4m3(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
    ) -> Result<(), GpuError> {
        if !self.modules.contains_key("f32_to_e4m3") {
            let module = self.compile_ptx(F32_TO_E4M3_PTX)?;
            self.modules.insert("f32_to_e4m3".to_string(), module);
        }

        let module = self.modules.get_mut("f32_to_e4m3").expect("just inserted");
        // Each thread processes 1 element
        let config = LaunchConfig::linear(count, 256);

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        // SAFETY: src_ptr and dst_ptr are valid GPU allocations
        unsafe {
            self.stream.launch_kernel(
                module,
                "f32_to_e4m3",
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

    /// PMAT-079: Convert FP16→FP32 with device-side activation dequant scaling.
    ///
    /// Reads act_dequant (act_absmax/448) from a device pointer and multiplies
    /// each converted element by it. Weight dequant is folded into GEMM alpha.
    fn convert_f16_to_f32_act_scaled(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
        act_dequant_ptr: u64,
    ) -> Result<(), GpuError> {
        let cache_key = "f16_to_f32_act_scaled";
        if !self.modules.contains_key(cache_key) {
            let module = self.compile_ptx(F16_TO_F32_ACT_SCALED_PTX)?;
            self.modules.insert(cache_key.to_string(), module);
        }

        let module = self.modules.get_mut(cache_key).expect("just inserted");
        let config = LaunchConfig::linear(count, 256);

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;
        let mut act_deq = act_dequant_ptr;

        unsafe {
            self.stream.launch_kernel(
                module,
                "f16_to_f32_act_scaled",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut act_deq) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-053b: Launch scaled FP32→FP8 E4M3 conversion kernel.
    /// Multiplies each element by quant_scale before E4M3 conversion.
    fn convert_f32_to_e4m3_scaled(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
        quant_scale: f32,
    ) -> Result<(), GpuError> {
        let cache_key = "f32_to_e4m3_scaled";
        if !self.modules.contains_key(cache_key) {
            let module = self.compile_ptx(F32_TO_E4M3_SCALED_PTX)?;
            self.modules.insert(cache_key.to_string(), module);
        }

        let module = self.modules.get_mut(cache_key).expect("just inserted");
        let num_blocks = (count + 255) / 256;
        let config = LaunchConfig {
            grid: (num_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        };

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;
        let mut scale = quant_scale;

        unsafe {
            self.stream.launch_kernel(
                module,
                "f32_to_e4m3_scaled",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut scale) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }

    /// PMAT-053b: Compute absmax of a GPU FP32 buffer via reduction kernel.
    /// Returns the absolute maximum value as f32.
    fn gpu_absmax(&mut self, src_ptr: u64, count: u32) -> Result<f32, GpuError> {
        if !self.modules.contains_key("absmax_reduce") {
            let module = self.compile_ptx(ABSMAX_REDUCE_PTX)?;
            self.modules.insert("absmax_reduce".to_string(), module);
        }

        // Allocate single u32 result buffer and zero it explicitly
        let mut result_buf = GpuBuffer::<u32>::new(&self.context, 1)?;
        let result_ptr = result_buf.as_ptr();
        // cuMemAlloc does NOT guarantee zeroed memory — zero explicitly
        result_buf.copy_from_host(&[0u32])?;

        let module = self
            .modules
            .get_mut("absmax_reduce")
            .expect("just inserted");
        let num_blocks = ((count + 255) / 256).min(256);
        let config = LaunchConfig {
            grid: (num_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem: 256 * 4,
        };

        let mut out = result_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        unsafe {
            self.stream.launch_kernel(
                module,
                "absmax_reduce",
                &config,
                &mut [
                    std::ptr::from_mut(&mut out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                ],
            )?;
        }

        self.stream.synchronize()?;

        // Read result back (u32 reinterpreted as f32)
        let mut result_u32 = [0u32; 1];
        result_buf.copy_to_host(&mut result_u32)?;
        let absmax = f32::from_bits(result_u32[0]);

        Ok(absmax)
    }

    /// PMAT-079: Compute absmax on GPU, keep result on device. No CPU sync.
    ///
    /// Returns the device pointer to the absmax result (u32 = f32 bits).
    /// The persistent buffer `fp8_absmax_buf` is reused across calls.
    fn gpu_absmax_device(&mut self, src_ptr: u64, count: u32) -> Result<u64, GpuError> {
        if !self.modules.contains_key("absmax_reduce") {
            let module = self.compile_ptx(ABSMAX_REDUCE_PTX)?;
            self.modules.insert("absmax_reduce".to_string(), module);
        }

        // Allocate or reuse persistent absmax buffer
        if self.fp8_absmax_buf.is_none() {
            self.fp8_absmax_buf = Some(GpuBuffer::<u32>::new(&self.context, 1)?);
        }
        let absmax_buf = self.fp8_absmax_buf.as_mut().expect("just allocated");
        let result_ptr = absmax_buf.as_ptr();
        // Zero the buffer on self.stream (NOT stream 0!) to avoid race with previous
        // matmul's conversion kernel that reads from this same buffer.
        // SAFETY: zero_val lives until stream.synchronize or kernel completion.
        // The zero is consumed by absmax_reduce (same stream, ordered after).
        let zero_val = [0u32; 1];
        unsafe {
            absmax_buf.copy_from_host_async(&zero_val, &self.stream)?;
        }

        let module = self
            .modules
            .get_mut("absmax_reduce")
            .expect("just inserted");
        let num_blocks = ((count + 255) / 256).min(256);
        let config = LaunchConfig {
            grid: (num_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem: 256 * 4,
        };

        let mut out = result_ptr;
        let mut src = src_ptr;
        let mut cnt = count;

        unsafe {
            self.stream.launch_kernel(
                module,
                "absmax_reduce",
                &config,
                &mut [
                    std::ptr::from_mut(&mut out) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // No sync! Result stays on device for the next kernel to read.
        Ok(result_ptr)
    }

    /// PMAT-079: Launch device-scaled FP32→FP8 E4M3 conversion kernel.
    ///
    /// Reads absmax from a device pointer (output of `gpu_absmax_device`).
    /// Also writes `absmax/448` to `dequant_ptr` for cuBLASLt A_SCALE_POINTER.
    fn convert_f32_to_e4m3_device_scaled(
        &mut self,
        src_ptr: u64,
        dst_ptr: u64,
        count: u32,
        absmax_ptr: u64,
        dequant_ptr: u64,
    ) -> Result<(), GpuError> {
        let cache_key = "f32_to_e4m3_device_scaled";
        if !self.modules.contains_key(cache_key) {
            let module = self.compile_ptx(F32_TO_E4M3_DEVICE_SCALED_PTX)?;
            self.modules.insert(cache_key.to_string(), module);
        }

        let module = self.modules.get_mut(cache_key).expect("just inserted");
        let num_blocks = (count + 255) / 256;
        let config = LaunchConfig {
            grid: (num_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        };

        let mut dst = dst_ptr;
        let mut src = src_ptr;
        let mut cnt = count;
        let mut abs_ptr = absmax_ptr;
        let mut deq_ptr = dequant_ptr;

        unsafe {
            self.stream.launch_kernel(
                module,
                "f32_to_e4m3_device_scaled",
                &config,
                &mut [
                    std::ptr::from_mut(&mut dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut cnt) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut abs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut deq_ptr) as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(())
    }
}
