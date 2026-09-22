
impl Default for CudaKernels {
    fn default() -> Self {
        Self::new()
    }
}

/// BUG-GGUF-001 FIX: Generate Q4_0 GEMV PTX with correct candle layout
///
/// The GGUF Q4_0 format uses "candle layout" where:
/// - 16 bytes contain 32 nibbles for 32 weights
/// - Low nibbles (byte & 0x0F) map to positions 0-15
/// - High nibbles (byte >> 4) map to positions 16-31
///
/// The trueno Q4_0GemvKernel incorrectly uses interleaved layout where:
/// - Thread 0 → byte 0 low nibble (position 0)
/// - Thread 1 → byte 0 high nibble (position 1)
/// - Thread 2 → byte 1 low nibble (position 2)
/// - etc.
///
/// This function generates correct PTX for GGUF Q4_0 models.
fn generate_q4_0_candle_ptx(k: u32, n: u32) -> String {
    // k and n are used for grid size configuration in the caller, not embedded in PTX
    let _ = (k, n);

    // Note: num_blocks is computed dynamically in PTX from k_dim parameter
    // This allows the same kernel to work for any K dimension
    String::from(
        r"
.version 7.5
.target sm_70
.address_size 64

// BUG-GGUF-001 FIX: Q4_0 GEMV with candle nibble layout
// Each warp (32 threads) computes one output element
// Thread 0-15: use low nibbles from bytes 0-15
// Thread 16-31: use high nibbles from bytes 0-15
.visible .entry q4_0_gemv_warp_reduce(
    .param .u64 y_ptr,
    .param .u64 w_ptr,
    .param .u64 x_ptr,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<16>;
    .reg .b16 %h<4>;
    .reg .pred %p<8>;

    // r0=tid, r1=ctaid, r2=n_dim, r3=k_dim
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;

    ld.param.u32 %r2, [n_dim];
    ld.param.u32 %r3, [k_dim];
    ld.param.u64 %rd0, [y_ptr];
    ld.param.u64 %rd1, [w_ptr];
    ld.param.u64 %rd2, [x_ptr];

    // Bounds check: if ctaid >= n_dim, exit
    setp.ge.u32 %p0, %r1, %r2;
    @%p0 bra $L_exit;

    // f0 = accumulator
    mov.f32 %f0, 0f00000000;

    // r4 = num_blocks = ceil(k_dim / 32)
    add.u32 %r4, %r3, 31;
    shr.u32 %r4, %r4, 5;

    // rd3 = row_base = w_ptr + ctaid * num_blocks * 18
    mul.lo.u32 %r5, %r4, 18;
    mul.wide.u32 %rd3, %r1, %r5;
    add.u64 %rd3, %rd1, %rd3;

    // r6 = blk_idx (loop counter)
    mov.u32 %r6, 0;

$L_blk_loop:
    setp.ge.u32 %p1, %r6, %r4;
    @%p1 bra $L_blk_loop_end;

    // rd4 = blk_addr = row_base + blk_idx * 18
    mul.wide.u32 %rd4, %r6, 18;
    add.u64 %rd4, %rd3, %rd4;

    // f1 = scale d (fp16 at offset 0) - use b16 register for f16 conversion
    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;

    // rd5 = qs_base = blk_addr + 2
    add.u64 %rd5, %rd4, 2;

    // CANDLE LAYOUT:
    // Thread 0-15 read bytes 0-15 (low nibbles -> positions 0-15)
    // Thread 16-31 read bytes 0-15 (high nibbles -> positions 16-31)
    // r8 = byte_idx = tid < 16 ? tid : tid - 16
    setp.ge.u32 %p2, %r0, 16;
    mov.u32 %r8, %r0;
    @%p2 sub.u32 %r8, %r0, 16;

    // Load byte from qs[byte_idx]
    cvt.u64.u32 %rd6, %r8;
    add.u64 %rd6, %rd5, %rd6;
    ld.global.u8 %r9, [%rd6];

    // r10 = nibble value
    // Threads 0-15: low nibble (byte & 0xF)
    // Threads 16-31: high nibble (byte >> 4)
    mov.u32 %r10, %r9;
    @%p2 shr.u32 %r10, %r9, 4;
    and.b32 %r10, %r10, 15;

    // r11 = centered value = nibble - 8 (as signed)
    sub.u32 %r11, %r10, 8;

    // f2 = dequantized = d * centered
    cvt.rn.f32.s32 %f2, %r11;
    mul.f32 %f2, %f1, %f2;

    // r12 = x_idx = blk_idx * 32 + tid
    shl.b32 %r12, %r6, 5;
    add.u32 %r12, %r12, %r0;

    // Bounds check for last block
    setp.ge.u32 %p3, %r12, %r3;
    @%p3 bra $L_skip_mul;

    // f3 = x[x_idx]
    cvt.u64.u32 %rd7, %r12;
    shl.b64 %rd7, %rd7, 2;
    add.u64 %rd7, %rd2, %rd7;
    ld.global.f32 %f3, [%rd7];

    // f0 += f2 * f3
    fma.rn.f32 %f0, %f2, %f3, %f0;

$L_skip_mul:
    add.u32 %r6, %r6, 1;
    bra $L_blk_loop;

$L_blk_loop_end:
    // Warp reduction using shfl.sync.down
    shfl.sync.down.b32 %f4, %f0, 16, 31, 0xffffffff;
    add.f32 %f0, %f0, %f4;
    shfl.sync.down.b32 %f5, %f0, 8, 31, 0xffffffff;
    add.f32 %f0, %f0, %f5;
    shfl.sync.down.b32 %f6, %f0, 4, 31, 0xffffffff;
    add.f32 %f0, %f0, %f6;
    shfl.sync.down.b32 %f7, %f0, 2, 31, 0xffffffff;
    add.f32 %f0, %f0, %f7;
    shfl.sync.down.b32 %f8, %f0, 1, 31, 0xffffffff;
    add.f32 %f0, %f0, %f8;

    // Thread 0 writes result
    setp.ne.u32 %p4, %r0, 0;
    @%p4 bra $L_exit;

    // y[ctaid] = f0
    mul.wide.u32 %rd8, %r1, 4;
    add.u64 %rd8, %rd0, %rd8;
    st.global.f32 [%rd8], %f0;

$L_exit:
    ret;
}
",
    )
}

/// PMAT-782 FIX: Generate Q4_1 GEMV PTX with correct candle (GGML) layout.
///
/// The GGUF Q4_1 format uses the same interleaved "candle layout" as Q4_0:
/// - 16 bytes contain 32 nibbles for 32 weights
/// - Low nibbles (byte & 0x0F) map to positions 0-15
/// - High nibbles (byte >> 4) map to positions 16-31
///
/// Q4_1 differs from Q4_0 only in dequantization: it is AFFINE, `val = d*nibble + m`
/// (a per-block fp16 min `m` at byte offset 2), with NO `- 8` centering. The block is
/// 20 bytes: `d` (fp16 @0) + `m` (fp16 @2) + 16 packed nibble bytes (@4).
///
/// The trueno `Q4_1GemvKernel` incorrectly used CONSECUTIVE layout (byte = tid/2,
/// nibble = tid&1), so every value index ≥1 mapped to the wrong nibble → garbage
/// logits for any Q4_1 tensor (e.g. Qwen2-0.5B FFN-down). This is the same defect
/// class BUG-GGUF-001/002 fixed for Q4_0/Q5_0; Q4_1 was simply never routed to a
/// candle generator. This function generates correct PTX for GGUF Q4_1 weights.
fn generate_q4_1_candle_ptx(k: u32, n: u32) -> String {
    // k and n are used for grid size configuration in the caller, not embedded in PTX
    let _ = (k, n);

    // Q4_1 block: 2 bytes (d fp16) + 2 bytes (m fp16) + 16 bytes (qs) = 20 bytes
    // Note: num_blocks is computed dynamically in PTX from k_dim parameter
    String::from(
        r"
.version 7.5
.target sm_70
.address_size 64

// PMAT-782 FIX: Q4_1 GEMV with candle nibble layout (affine dequant d*q + m)
// Each warp (32 threads) computes one output element
// Thread 0-15: use low nibbles from bytes 0-15
// Thread 16-31: use high nibbles from bytes 0-15
.visible .entry q4_1_gemv_warp_reduce(
    .param .u64 y_ptr,
    .param .u64 w_ptr,
    .param .u64 x_ptr,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<16>;
    .reg .b16 %h<4>;
    .reg .pred %p<8>;

    // r0=tid, r1=ctaid, r2=n_dim, r3=k_dim
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;

    ld.param.u32 %r2, [n_dim];
    ld.param.u32 %r3, [k_dim];
    ld.param.u64 %rd0, [y_ptr];
    ld.param.u64 %rd1, [w_ptr];
    ld.param.u64 %rd2, [x_ptr];

    // Bounds check: if ctaid >= n_dim, exit
    setp.ge.u32 %p0, %r1, %r2;
    @%p0 bra $L_exit;

    // f0 = accumulator
    mov.f32 %f0, 0f00000000;

    // r4 = num_blocks = ceil(k_dim / 32)
    add.u32 %r4, %r3, 31;
    shr.u32 %r4, %r4, 5;

    // rd3 = row_base = w_ptr + ctaid * num_blocks * 20
    mul.lo.u32 %r5, %r4, 20;
    mul.wide.u32 %rd3, %r1, %r5;
    add.u64 %rd3, %rd1, %rd3;

    // r6 = blk_idx (loop counter)
    mov.u32 %r6, 0;

$L_blk_loop:
    setp.ge.u32 %p1, %r6, %r4;
    @%p1 bra $L_blk_loop_end;

    // rd4 = blk_addr = row_base + blk_idx * 20
    mul.wide.u32 %rd4, %r6, 20;
    add.u64 %rd4, %rd3, %rd4;

    // f1 = scale d (fp16 at offset 0)
    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;

    // f4 = min m (fp16 at offset 2)
    add.u64 %rd5, %rd4, 2;
    ld.global.b16 %h1, [%rd5];
    cvt.f32.f16 %f4, %h1;

    // rd6 = qs_base = blk_addr + 4
    add.u64 %rd6, %rd4, 4;

    // CANDLE LAYOUT:
    // Thread 0-15 read bytes 0-15 (low nibbles -> positions 0-15)
    // Thread 16-31 read bytes 0-15 (high nibbles -> positions 16-31)
    // r8 = byte_idx = tid < 16 ? tid : tid - 16
    setp.ge.u32 %p2, %r0, 16;
    mov.u32 %r8, %r0;
    @%p2 sub.u32 %r8, %r0, 16;

    // Load byte from qs[byte_idx]
    cvt.u64.u32 %rd7, %r8;
    add.u64 %rd7, %rd6, %rd7;
    ld.global.u8 %r9, [%rd7];

    // r10 = nibble value
    // Threads 0-15: low nibble (byte & 0xF)
    // Threads 16-31: high nibble (byte >> 4)
    mov.u32 %r10, %r9;
    @%p2 shr.u32 %r10, %r9, 4;
    and.b32 %r10, %r10, 15;

    // Q4_1 affine dequant: f2 = d * nibble + m (no centering)
    cvt.rn.f32.u32 %f2, %r10;
    fma.rn.f32 %f2, %f1, %f2, %f4;

    // r12 = x_idx = blk_idx * 32 + tid
    shl.b32 %r12, %r6, 5;
    add.u32 %r12, %r12, %r0;

    // Bounds check for last block
    setp.ge.u32 %p3, %r12, %r3;
    @%p3 bra $L_skip_mul;

    // f3 = x[x_idx]
    cvt.u64.u32 %rd8, %r12;
    shl.b64 %rd8, %rd8, 2;
    add.u64 %rd8, %rd2, %rd8;
    ld.global.f32 %f3, [%rd8];

    // f0 += f2 * f3
    fma.rn.f32 %f0, %f2, %f3, %f0;

$L_skip_mul:
    add.u32 %r6, %r6, 1;
    bra $L_blk_loop;

$L_blk_loop_end:
    // Warp reduction using shfl.sync.down
    shfl.sync.down.b32 %f5, %f0, 16, 31, 0xffffffff;
    add.f32 %f0, %f0, %f5;
    shfl.sync.down.b32 %f6, %f0, 8, 31, 0xffffffff;
    add.f32 %f0, %f0, %f6;
    shfl.sync.down.b32 %f7, %f0, 4, 31, 0xffffffff;
    add.f32 %f0, %f0, %f7;
    shfl.sync.down.b32 %f8, %f0, 2, 31, 0xffffffff;
    add.f32 %f0, %f0, %f8;
    shfl.sync.down.b32 %f9, %f0, 1, 31, 0xffffffff;
    add.f32 %f0, %f0, %f9;

    // Thread 0 writes result
    setp.ne.u32 %p4, %r0, 0;
    @%p4 bra $L_exit;

    // y[ctaid] = f0
    mul.wide.u32 %rd9, %r1, 4;
    add.u64 %rd9, %rd0, %rd9;
    st.global.f32 [%rd9], %f0;

$L_exit:
    ret;
}
",
    )
}

/// #3477 / "no model left behind": IQ4_XS (GGML type 23) GEMV, row-major.
///
/// `Qwen3.5-4B-UD-Q4_K_XL` stores `ffn_gate`/`ffn_up` in blk.12, 13, 16 … as
/// IQ4_XS. These are the large tensors — `[2560, 9216]` — so this is the kernel
/// that decides whether the model generates, not merely whether it loads.
///
/// PORT OF A VERIFIED REFERENCE, not a fresh transcription. The layout below is
/// `quantize::iq4_xs::dequantize_iq4_xs_block`, which its own header records as
/// transcribed from llama.cpp `ggml-quants.c` @ df03399 and independently
/// confirmed against gguf-py's numpy dequantizers at 50 random blocks per type.
/// Any disagreement between this kernel and that function is this kernel's bug.
///
/// Super-block: 136 bytes / 256 elements (4.25 bits per weight).
/// ```text
///   [0..2)    d         f16 super-block scale
///   [2..4)    scales_h  u16, two high bits of each sub-block scale
///   [4..8)    scales_l  4 bytes, four bits of each sub-block scale
///   [8..136)  qs        128 bytes = 8 sub-blocks x 16 bytes, two 4-bit
///                       indices per byte into the 16 non-linear IQ4_NL levels
/// ```
/// Per sub-block `ib`: `ls = (scales_l[ib/2] >> (4*(ib%2))) & 0xf | ((scales_h >> (2*ib)) & 3) << 4`
/// (6 bits), then `dl = d * (ls - 32)`, and element `j` of the low half is
/// `dl * KVALUES_IQ4NL[byte & 0xf]`, the high half `dl * KVALUES_IQ4NL[byte >> 4]`.
///
/// THREAD MAPPING. One warp per output row. Element `e = tid + 32*m` for
/// `m` in 0..8, and because `tid < 32` this gives `ib == m` and `jj == tid`
/// exactly — so each thread handles precisely one element of each sub-block and
/// the per-sub-block scale is computed once per iteration rather than per
/// element. `tid >> 4` selects the high or low nibble; `tid & 15` is the byte.
///
/// LAYOUT-001: row `i` starts at `w_ptr + i * nb * 136` and runs contiguously.
/// Row-major, no transpose, no `*_colmajor` kernel in reach.
///
/// ALIGNMENT: 136 is even and `w_ptr` is device-aligned, so every block address
/// is even and the two 16-bit loads (`d` at +0, `scales_h` at +2) are aligned.
/// `scales_l` and `qs` are read as bytes, which needs no alignment at all.
fn generate_iq4_xs_gemv_ptx(k: u32, n: u32) -> String {
    let _ = (k, n);

    String::from(
        r"
.version 7.5
.target sm_70
.address_size 64

// The 16 non-linear IQ4_NL levels: quantize::iq_grids::KVALUES_IQ4NL.
.global .align 4 .s32 kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

.visible .entry iq4_xs_gemv_warp_reduce(
    .param .u64 y_ptr,
    .param .u64 w_ptr,
    .param .u64 x_ptr,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .reg .u32 %r<48>;
    .reg .u64 %rd<32>;
    .reg .f32 %f<24>;
    .reg .b16 %h<4>;
    .reg .pred %p<12>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;

    ld.param.u32 %r2, [n_dim];
    ld.param.u32 %r3, [k_dim];
    ld.param.u64 %rd0, [y_ptr];
    ld.param.u64 %rd1, [w_ptr];
    ld.param.u64 %rd2, [x_ptr];

    setp.ge.u32 %p0, %r1, %r2;
    @%p0 bra $L_iq_exit;

    mov.f32 %f0, 0f00000000;

    // nb = ceil(k_dim / 256)
    add.u32 %r4, %r3, 255;
    shr.u32 %r4, %r4, 8;

    // row_base = w_ptr + ctaid * nb * 136
    mul.lo.u32 %r5, %r4, 136;
    mul.wide.u32 %rd3, %r1, %r5;
    add.u64 %rd3, %rd1, %rd3;

    // per-thread: jhalf = tid >> 4 (nibble select), jlow = tid & 15 (byte index)
    shr.u32 %r6, %r0, 4;
    and.b32 %r7, %r0, 15;

    // codebook base
    mov.u64 %rd10, kvalues_iq4nl;

    mov.u32 %r8, 0;                 // blk

$L_iq_blk:
    setp.ge.u32 %p1, %r8, %r4;
    @%p1 bra $L_iq_blk_end;

    // blk_addr = row_base + blk * 136
    mul.wide.u32 %rd4, %r8, 136;
    add.u64 %rd4, %rd3, %rd4;

    // d (f16 at +0)
    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;

    // scales_h (u16 at +2)
    add.u64 %rd5, %rd4, 2;
    ld.global.u16 %r9, [%rd5];

    mov.u32 %r10, 0;                // m = sub-block index = ib

$L_iq_sub:
    setp.ge.u32 %p2, %r10, 8;
    @%p2 bra $L_iq_sub_end;

    // ls_low = (scales_l[m >> 1] >> (4 * (m & 1))) & 0xf
    shr.u32 %r11, %r10, 1;
    cvt.u64.u32 %rd6, %r11;
    add.u64 %rd6, %rd4, %rd6;
    add.u64 %rd6, %rd6, 4;
    ld.global.u8 %r12, [%rd6];
    and.b32 %r13, %r10, 1;
    shl.b32 %r13, %r13, 2;
    shr.u32 %r12, %r12, %r13;
    and.b32 %r12, %r12, 15;

    // ls_high = ((scales_h >> (2 * m)) & 3) << 4
    shl.b32 %r14, %r10, 1;
    shr.u32 %r15, %r9, %r14;
    and.b32 %r15, %r15, 3;
    shl.b32 %r15, %r15, 4;

    // ls = ls_low | ls_high ; dl = d * (ls - 32)
    or.b32 %r16, %r12, %r15;
    cvt.rn.f32.u32 %f2, %r16;
    sub.f32 %f2, %f2, 0f42000000;   // 32.0
    mul.f32 %f3, %f1, %f2;

    // byte = qs[16*m + jlow]  (qs starts at +8)
    shl.b32 %r17, %r10, 4;
    add.u32 %r17, %r17, %r7;
    cvt.u64.u32 %rd7, %r17;
    add.u64 %rd7, %rd4, %rd7;
    add.u64 %rd7, %rd7, 8;
    ld.global.u8 %r18, [%rd7];

    // nib = jhalf ? (byte >> 4) : (byte & 0xf)
    shl.b32 %r19, %r6, 2;           // jhalf * 4
    shr.u32 %r20, %r18, %r19;
    and.b32 %r20, %r20, 15;

    // w = dl * kvalues_iq4nl[nib]
    mul.wide.u32 %rd8, %r20, 4;
    add.u64 %rd8, %rd10, %rd8;
    ld.global.s32 %r21, [%rd8];
    cvt.rn.f32.s32 %f4, %r21;
    mul.f32 %f5, %f3, %f4;

    // x_idx = blk*256 + m*32 + tid
    shl.b32 %r22, %r8, 8;
    shl.b32 %r23, %r10, 5;
    add.u32 %r22, %r22, %r23;
    add.u32 %r22, %r22, %r0;

    setp.ge.u32 %p3, %r22, %r3;
    @%p3 bra $L_iq_skip;

    mul.wide.u32 %rd9, %r22, 4;
    add.u64 %rd9, %rd2, %rd9;
    ld.global.f32 %f6, [%rd9];
    fma.rn.f32 %f0, %f5, %f6, %f0;

$L_iq_skip:
    add.u32 %r10, %r10, 1;
    bra $L_iq_sub;

$L_iq_sub_end:
    add.u32 %r8, %r8, 1;
    bra $L_iq_blk;

$L_iq_blk_end:
    shfl.sync.down.b32 %f10, %f0, 16, 31, 0xffffffff;
    add.f32 %f0, %f0, %f10;
    shfl.sync.down.b32 %f11, %f0, 8, 31, 0xffffffff;
    add.f32 %f0, %f0, %f11;
    shfl.sync.down.b32 %f12, %f0, 4, 31, 0xffffffff;
    add.f32 %f0, %f0, %f12;
    shfl.sync.down.b32 %f13, %f0, 2, 31, 0xffffffff;
    add.f32 %f0, %f0, %f13;
    shfl.sync.down.b32 %f14, %f0, 1, 31, 0xffffffff;
    add.f32 %f0, %f0, %f14;

    setp.ne.u32 %p4, %r0, 0;
    @%p4 bra $L_iq_exit;

    mul.wide.u32 %rd11, %r1, 4;
    add.u64 %rd11, %rd0, %rd11;
    st.global.f32 [%rd11], %f0;

$L_iq_exit:
    ret;
}
",
    )
}

/// #3477 / "no model left behind": F16 (GGML type 1) GEMV, row-major.
///
/// `Qwen3.5-4B-UD-Q4_K_XL` stores `ssm_alpha` and `ssm_beta` as F16 while the
/// rest of the model is Q4_K/Q5_K/IQ4_XS. Before this kernel existed,
/// `WeightQuantType::from_ggml_type(1)` returned `None`, so the hybrid refused
/// the whole model to CPU — `hybrid_gpu_unsupported_quant_tensor` names
/// `blk.0.ssm_alpha` first.
///
/// LAYOUT-001: GGUF/APR are ROW-MAJOR here. Row `ctaid` of an `[n, k]` weight
/// starts at `w_ptr + ctaid * k * 2` bytes and runs contiguously, exactly as
/// the block-quantized kernels beside this one index their rows. There is no
/// transpose and no `*_colmajor` path — those are forbidden for GGUF data.
///
/// One warp per output row, each thread striding the row by 32, `cvt.f32.f16`
/// on the load, then the same `shfl.sync.down` reduction every GEMV in this
/// file uses. F16 is unquantized, so there are no blocks, no scales and no
/// codebook: this is the F32 kernel with a converting load and a 2-byte stride.
fn generate_f16_gemv_ptx(k: u32, n: u32) -> String {
    // k and n size the grid in the caller; the PTX reads them as parameters so
    // one module serves every shape (the cache key still carries them).
    let _ = (k, n);

    String::from(
        r"
.version 7.5
.target sm_70
.address_size 64

// F16 GEMV, row-major: y[row] = sum_i f16_to_f32(w[row*k + i]) * x[i]
.visible .entry f16_gemv_warp_reduce(
    .param .u64 y_ptr,
    .param .u64 w_ptr,
    .param .u64 x_ptr,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .reg .u32 %r<20>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<16>;
    .reg .b16 %h<4>;
    .reg .pred %p<8>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;

    ld.param.u32 %r2, [n_dim];
    ld.param.u32 %r3, [k_dim];
    ld.param.u64 %rd0, [y_ptr];
    ld.param.u64 %rd1, [w_ptr];
    ld.param.u64 %rd2, [x_ptr];

    // Rows beyond n_dim do no work (grid is padded to whole warps).
    setp.ge.u32 %p0, %r1, %r2;
    @%p0 bra $L_exit;

    mov.f32 %f0, 0f00000000;

    // rd3 = row_base = w_ptr + ctaid * k_dim * 2   (2 bytes per f16)
    shl.b32 %r4, %r3, 1;
    mul.wide.u32 %rd3, %r1, %r4;
    add.u64 %rd3, %rd1, %rd3;

    // i = tid, stride 32
    mov.u32 %r5, %r0;

$L_loop:
    setp.ge.u32 %p1, %r5, %r3;
    @%p1 bra $L_loop_end;

    // w = f16_to_f32(row_base[i])
    mul.wide.u32 %rd4, %r5, 2;
    add.u64 %rd4, %rd3, %rd4;
    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;

    // x = x_ptr[i]
    mul.wide.u32 %rd5, %r5, 4;
    add.u64 %rd5, %rd2, %rd5;
    ld.global.f32 %f2, [%rd5];

    fma.rn.f32 %f0, %f1, %f2, %f0;

    add.u32 %r5, %r5, 32;
    bra $L_loop;

$L_loop_end:
    // Warp reduction: identical idiom to the block-quantized GEMVs here.
    shfl.sync.down.b32 %f4, %f0, 16, 31, 0xffffffff;
    add.f32 %f0, %f0, %f4;
    shfl.sync.down.b32 %f5, %f0, 8, 31, 0xffffffff;
    add.f32 %f0, %f0, %f5;
    shfl.sync.down.b32 %f6, %f0, 4, 31, 0xffffffff;
    add.f32 %f0, %f0, %f6;
    shfl.sync.down.b32 %f7, %f0, 2, 31, 0xffffffff;
    add.f32 %f0, %f0, %f7;
    shfl.sync.down.b32 %f8, %f0, 1, 31, 0xffffffff;
    add.f32 %f0, %f0, %f8;

    setp.ne.u32 %p2, %r0, 0;
    @%p2 bra $L_exit;

    mul.wide.u32 %rd6, %r1, 4;
    add.u64 %rd6, %rd0, %rd6;
    st.global.f32 [%rd6], %f0;

$L_exit:
    ret;
}
",
    )
}

/// BUG-GGUF-002 FIX: Generate Q5_0 GEMV PTX with correct candle layout
///
/// The GGUF Q5_0 format uses "candle layout" where:
/// - 16 bytes contain 32 nibbles for 32 weights (low bits)
/// - 4 bytes contain 32 high bits (qh)
/// - Low nibbles (byte & 0x0F) + qh bits 0-15 map to positions 0-15
/// - High nibbles (byte >> 4) + qh bits 16-31 map to positions 16-31
///
/// The trueno Q5_0GemvKernel incorrectly uses interleaved layout where:
/// - Thread 0 → byte 0 low nibble + qh bit 0 (position 0)
/// - Thread 1 → byte 0 high nibble + qh bit 1 (position 1)
/// - Thread 2 → byte 1 low nibble + qh bit 2 (position 2)
/// - etc.
///
/// This function generates correct PTX for GGUF Q5_0 models.
fn generate_q5_0_candle_ptx(k: u32, n: u32) -> String {
    // k and n are used for grid size configuration in the caller, not embedded in PTX
    let _ = (k, n);

    // Q5_0 block: 2 bytes (d fp16) + 4 bytes (qh) + 16 bytes (qs) = 22 bytes
    // Note: num_blocks is computed dynamically in PTX from k_dim parameter
    String::from(
        r"
.version 7.5
.target sm_70
.address_size 64

// BUG-GGUF-002 FIX: Q5_0 GEMV with candle nibble layout
// Each warp (32 threads) computes one output element
// Thread 0-15: use low nibbles from bytes 0-15, qh bits 0-15
// Thread 16-31: use high nibbles from bytes 0-15, qh bits 16-31
.visible .entry q5_0_gemv_warp_reduce(
    .param .u64 y_ptr,
    .param .u64 w_ptr,
    .param .u64 x_ptr,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .reg .u32 %r<40>;
    .reg .u64 %rd<20>;
    .reg .f32 %f<16>;
    .reg .b16 %h<4>;
    .reg .pred %p<8>;

    // r0=tid, r1=ctaid, r2=n_dim, r3=k_dim
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;

    ld.param.u32 %r2, [n_dim];
    ld.param.u32 %r3, [k_dim];
    ld.param.u64 %rd0, [y_ptr];
    ld.param.u64 %rd1, [w_ptr];
    ld.param.u64 %rd2, [x_ptr];

    // Bounds check: if ctaid >= n_dim, exit
    setp.ge.u32 %p0, %r1, %r2;
    @%p0 bra $L_exit;

    // f0 = accumulator
    mov.f32 %f0, 0f00000000;

    // r4 = num_blocks = ceil(k_dim / 32)
    add.u32 %r4, %r3, 31;
    shr.u32 %r4, %r4, 5;

    // rd3 = row_base = w_ptr + ctaid * num_blocks * 22
    mul.lo.u32 %r5, %r4, 22;
    mul.wide.u32 %rd3, %r1, %r5;
    add.u64 %rd3, %rd1, %rd3;

    // r6 = blk_idx (loop counter)
    mov.u32 %r6, 0;

$L_blk_loop:
    setp.ge.u32 %p1, %r6, %r4;
    @%p1 bra $L_blk_loop_end;

    // rd4 = blk_addr = row_base + blk_idx * 22
    mul.wide.u32 %rd4, %r6, 22;
    add.u64 %rd4, %rd3, %rd4;

    // f1 = scale d (fp16 at offset 0) - use b16 register for f16 conversion
    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;

    // Load qh (4 bytes at offset 2) using byte loads for unaligned access
    add.u64 %rd5, %rd4, 2;
    ld.global.u8 %r20, [%rd5];
    add.u64 %rd6, %rd4, 3;
    ld.global.u8 %r21, [%rd6];
    add.u64 %rd7, %rd4, 4;
    ld.global.u8 %r22, [%rd7];
    add.u64 %rd8, %rd4, 5;
    ld.global.u8 %r23, [%rd8];
    // Combine: qh = r20 | (r21 << 8) | (r22 << 16) | (r23 << 24)
    shl.b32 %r24, %r21, 8;
    shl.b32 %r25, %r22, 16;
    shl.b32 %r26, %r23, 24;
    or.b32 %r27, %r20, %r24;
    or.b32 %r28, %r27, %r25;
    or.b32 %r8, %r28, %r26;  // r8 = qh

    // rd9 = qs_base = blk_addr + 6
    add.u64 %rd9, %rd4, 6;

    // CANDLE LAYOUT:
    // Thread 0-15 read bytes 0-15 (low nibbles -> positions 0-15), qh bits 0-15
    // Thread 16-31 read bytes 0-15 (high nibbles -> positions 16-31), qh bits 16-31
    // r9 = byte_idx = tid < 16 ? tid : tid - 16
    setp.ge.u32 %p2, %r0, 16;
    mov.u32 %r9, %r0;
    @%p2 sub.u32 %r9, %r0, 16;

    // Load byte from qs[byte_idx]
    cvt.u64.u32 %rd10, %r9;
    add.u64 %rd10, %rd9, %rd10;
    ld.global.u8 %r10, [%rd10];

    // r11 = nibble value
    // Threads 0-15: low nibble (byte & 0xF)
    // Threads 16-31: high nibble (byte >> 4)
    mov.u32 %r11, %r10;
    @%p2 shr.u32 %r11, %r10, 4;
    and.b32 %r11, %r11, 15;

    // Extract high bit: (qh >> tid) & 1
    // For candle layout, threads 0-15 use qh bits 0-15, threads 16-31 use qh bits 16-31
    shr.b32 %r12, %r8, %r0;
    and.b32 %r12, %r12, 1;

    // Combine: q5 = nibble | (high_bit << 4)
    shl.b32 %r13, %r12, 4;
    or.b32 %r14, %r11, %r13;

    // r15 = centered value = q5 - 16 (as signed)
    sub.u32 %r15, %r14, 16;

    // f2 = dequantized = d * centered
    cvt.rn.f32.s32 %f2, %r15;
    mul.f32 %f2, %f1, %f2;

    // r16 = x_idx = blk_idx * 32 + tid
    shl.b32 %r16, %r6, 5;
    add.u32 %r16, %r16, %r0;

    // Bounds check for last block
    setp.ge.u32 %p3, %r16, %r3;
    @%p3 bra $L_skip_mul;

    // f3 = x[x_idx]
    cvt.u64.u32 %rd11, %r16;
    shl.b64 %rd11, %rd11, 2;
    add.u64 %rd11, %rd2, %rd11;
    ld.global.f32 %f3, [%rd11];

    // f0 += f2 * f3
    fma.rn.f32 %f0, %f2, %f3, %f0;

$L_skip_mul:
    add.u32 %r6, %r6, 1;
    bra $L_blk_loop;

$L_blk_loop_end:
    // Warp reduction using shfl.sync.down
    shfl.sync.down.b32 %f4, %f0, 16, 31, 0xffffffff;
    add.f32 %f0, %f0, %f4;
    shfl.sync.down.b32 %f5, %f0, 8, 31, 0xffffffff;
    add.f32 %f0, %f0, %f5;
    shfl.sync.down.b32 %f6, %f0, 4, 31, 0xffffffff;
    add.f32 %f0, %f0, %f6;
    shfl.sync.down.b32 %f7, %f0, 2, 31, 0xffffffff;
    add.f32 %f0, %f0, %f7;
    shfl.sync.down.b32 %f8, %f0, 1, 31, 0xffffffff;
    add.f32 %f0, %f0, %f8;

    // Thread 0 writes result
    setp.ne.u32 %p4, %r0, 0;
    @%p4 bra $L_exit;

    // y[ctaid] = f0
    mul.wide.u32 %rd12, %r1, 4;
    add.u64 %rd12, %rd0, %rd12;
    st.global.f32 [%rd12], %f0;

$L_exit:
    ret;
}
",
    )
}
