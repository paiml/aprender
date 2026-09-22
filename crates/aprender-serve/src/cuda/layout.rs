
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

/// #3885: Q5_1 (GGML type 7) GEMV, row-major.
///
/// The simplest kernel in this file and the last blocker on
/// `Qwen2.5-0.5B-Instruct-IQ4_XS`, which carries 24 Q5_1 tensors among 290.
/// A legacy 32-element block in 24 bytes: `d` f16 at +0, `m` f16 at +2, `qh`
/// u32 at +4, `qs[16]` at +8. No codebook, no sign bytes, no sub-block scales.
///
/// The one subtlety is the 5th bit, which is what distinguishes Q5_1 from Q4_1:
/// the low nibble of byte `j` takes bit `j` of `qh` and the high nibble takes
/// bit `j + 16`, each contributing 16 to the quantized value. Both nibbles of a
/// byte are 16 elements apart in the output, NOT adjacent - the reference's
/// `y[j]` / `y[j + QK5_1/2]` split, the same shape as IQ4_NL's.
///
/// Q5_1 is AFFINE, unlike every other kernel here: `w = q * d + m`. The min is
/// per-block and added after scaling, so a kernel that drops it produces values
/// with the right spread and the wrong centre.
///
/// LAYOUT-001: row-major. Row `ctaid` starts at `w_ptr + ctaid * ceil(k/32) * 24`.
fn generate_q5_1_gemv_ptx(k: u32, n: u32) -> String {
    let _ = (k, n);

    String::from(
        r"
.version 7.5
.target sm_70
.address_size 64

.visible .entry q5_1_gemv_warp_reduce(
    .param .u64 y_ptr,
    .param .u64 w_ptr,
    .param .u64 x_ptr,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .reg .u32 %r<40>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<20>;
    .reg .b16 %h<4>;
    .reg .pred %p<10>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;

    ld.param.u32 %r2, [n_dim];
    ld.param.u32 %r3, [k_dim];
    ld.param.u64 %rd0, [y_ptr];
    ld.param.u64 %rd1, [w_ptr];
    ld.param.u64 %rd2, [x_ptr];

    setp.ge.u32 %p0, %r1, %r2;
    @%p0 bra $L_q51_exit;

    mov.f32 %f0, 0f00000000;

    // nb = ceil(k_dim / 32)
    add.u32 %r4, %r3, 31;
    shr.u32 %r4, %r4, 5;

    // row_base = w_ptr + ctaid * nb * 24
    mul.lo.u32 %r5, %r4, 24;
    mul.wide.u32 %rd3, %r1, %r5;
    add.u64 %rd3, %rd1, %rd3;

    // jlow = tid & 15 (byte index), jhalf = tid >> 4 (nibble select)
    and.b32 %r7, %r0, 15;
    shr.u32 %r6, %r0, 4;

    // nibble shift = jhalf * 4
    shl.b32 %r19, %r6, 2;
    // qh bit index = jlow + 16*jhalf
    shl.b32 %r23, %r6, 4;
    add.u32 %r23, %r23, %r7;

    mov.u32 %r8, 0;

$L_q51_blk:
    setp.ge.u32 %p1, %r8, %r4;
    @%p1 bra $L_q51_blk_end;

    // blk_addr = row_base + blk * 24
    mul.wide.u32 %rd4, %r8, 24;
    add.u64 %rd4, %rd3, %rd4;

    // d (f16 at +0), m (f16 at +2)
    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;
    ld.global.b16 %h1, [%rd4+2];
    cvt.f32.f16 %f2, %h1;

    // qh (u32 at +4)
    ld.global.u32 %r24, [%rd4+4];

    // byte = qs[jlow], qs starts at +8
    cvt.u64.u32 %rd7, %r7;
    add.u64 %rd7, %rd4, %rd7;
    add.u64 %rd7, %rd7, 8;
    ld.global.u8 %r18, [%rd7];

    // nib = (byte >> (4*jhalf)) & 0xf
    shr.u32 %r20, %r18, %r19;
    and.b32 %r20, %r20, 15;

    // hb = (qh >> (jlow + 16*jhalf)) & 1, placed at bit 4
    shr.u32 %r25, %r24, %r23;
    and.b32 %r25, %r25, 1;
    shl.b32 %r25, %r25, 4;
    or.b32 %r20, %r20, %r25;

    // w = q * d + m
    cvt.rn.f32.u32 %f4, %r20;
    fma.rn.f32 %f5, %f4, %f1, %f2;

    // x_idx = blk*32 + tid
    shl.b32 %r22, %r8, 5;
    add.u32 %r22, %r22, %r0;

    setp.ge.u32 %p3, %r22, %r3;
    @%p3 bra $L_q51_skip;

    mul.wide.u32 %rd9, %r22, 4;
    add.u64 %rd9, %rd2, %rd9;
    ld.global.f32 %f6, [%rd9];
    fma.rn.f32 %f0, %f5, %f6, %f0;

$L_q51_skip:
    add.u32 %r8, %r8, 1;
    bra $L_q51_blk;

$L_q51_blk_end:
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
    @%p4 bra $L_q51_exit;

    mul.wide.u32 %rd11, %r1, 4;
    add.u64 %rd11, %rd0, %rd11;
    st.global.f32 [%rd11], %f0;

$L_q51_exit:
    ret;
}
",
    )
}

/// #3884: IQ3_S (GGML type 21) GEMV, row-major.
///
/// The most intricate codebook type here: 3.4375 bits/weight as 9-bit indices
/// into a 512-entry grid (8 bits in `qs`, the 9th in `qh`), explicit sign bytes,
/// and 4-bit scales each shared by two sub-blocks. 110 bytes per 256-element
/// super-block: `d` f16 at +0, `qs[64]` at +2, `qh[8]` at +66, `signs[32]` at
/// +74, `scales[4]` at +106.
///
/// THREAD MAPPING: 8 sub-blocks x 4 groups is exactly 32, so `ib = tid >> 2` and
/// `l = tid & 3` gives one warp lane per (sub-block, group) pair, each owning 8
/// consecutive outputs at `32*ib + 8*l`. No lane does two groups and none idles.
///
/// Two simplifications over the reference, both verified equivalent rather than
/// assumed. `KMASK_IQ2XS` is `[1,2,4,8,16,32,64,128]`, so `sign_byte & KMASK[j]`
/// is just bit `j`. And ggml's `(qh << (8-2l)) & 256` / `(qh << (7-2l)) & 256`
/// select bits `2l` and `2l+1` of `qh`, so they become `((qh >> 2l) & 1) << 8`.
///
/// The scale pairing is the part a port gets wrong: the reference walks
/// `ib32` in steps of 2 with an inner `half`, indexing `scales[ib32/2]` and
/// taking the low nibble for `half == 0`. For a flat `ib` that is
/// `scales[ib >> 1]`, low nibble when `ib & 1 == 0` — the same byte serves two
/// consecutive sub-blocks.
///
/// LAYOUT-001: row-major. Row `ctaid` starts at `w_ptr + ctaid * ceil(k/256) * 110`.
fn generate_iq3_s_gemv_ptx(k: u32, n: u32) -> String {
    let _ = (k, n);

    String::from(
        r"
.version 7.5
.target sm_70
.address_size 64

// IQ3S_GRID: quantize::iq_grids::IQ3S_GRID, 512 packed 4-byte codebook entries.
.global .align 4 .u32 iq3s_grid_g[512] = {
    16843009, 16843011, 16843013, 16843019, 16843023, 16843521, 16843523, 16843525,
    16843529, 16843533, 16844033, 16844035, 16844043, 16844551, 16845057, 16845061,
    16845067, 16845071, 16845571, 16845575, 16846081, 16846085, 16846595, 16846601,
    16846607, 16974081, 16974083, 16974085, 16974089, 16974593, 16974595, 16974603,
    16975105, 16975111, 16975119, 16975619, 16975627, 16976137, 16977155, 16977163,
    16977669, 17105153, 17105155, 17105163, 17105167, 17105665, 17105671, 17105677,
    17106179, 17106187, 17106689, 17106697, 17107205, 17107211, 17107215, 17107715,
    17107719, 17108737, 17108743, 17236231, 17236739, 17236747, 17237249, 17237253,
    17237763, 17237767, 17237773, 17238281, 17238785, 17238789, 17239311, 17239811,
    17239819, 17367297, 17367815, 17367823, 17368323, 17368329, 17368837, 17369345,
    17369351, 17369859, 17370881, 17498373, 17498377, 17499393, 17499397, 17499405,
    17499911, 17500419, 17500427, 17500431, 17501453, 17501959, 17629453, 17629955,
    17629959, 17630979, 17632005, 17633027, 17760513, 17760517, 17760521, 17761537,
    17761541, 17761549, 17762055, 17763073, 17763081, 50397441, 50397443, 50397445,
    50397449, 50397953, 50397955, 50397959, 50397963, 50397967, 50398465, 50398469,
    50398979, 50398985, 50398989, 50400009, 50400013, 50400515, 50401029, 50528513,
    50528515, 50528519, 50528525, 50529025, 50529033, 50529539, 50530049, 50530055,
    50530563, 50531073, 50531077, 50532097, 50532109, 50659585, 50660101, 50660107,
    50660111, 50660609, 50660617, 50661125, 50661633, 50661639, 50662155, 50662657,
    50663173, 50790659, 50790665, 50790671, 50791169, 50791175, 50791683, 50791695,
    50792193, 50792201, 50792707, 50793733, 50794241, 50921735, 50921739, 50922245,
    50922249, 50923267, 50923271, 50923781, 50923789, 50924289, 50924297, 51052803,
    51053313, 51053319, 51053827, 51054337, 51054341, 51055363, 51184897, 51184905,
    51184911, 51185929, 51185933, 51314947, 51314951, 51315457, 51315461, 51315971,
    51316491, 51316995, 51318021, 51318529, 83951873, 83951875, 83951879, 83951883,
    83951887, 83952385, 83952389, 83952393, 83952397, 83952899, 83952903, 83952911,
    83953409, 83953413, 83953923, 83953927, 83953931, 83954433, 83954437, 83954959,
    83955457, 83955463, 83955467, 84082945, 84082949, 84083457, 84083463, 84083471,
    84083973, 84083979, 84084483, 84084489, 84084997, 84085507, 84214019, 84214025,
    84214031, 84215043, 84215047, 84215553, 84215567, 84216067, 84216583, 84216591,
    84217603, 84217609, 84345089, 84345093, 84345099, 84345603, 84346117, 84346121,
    84346627, 84346631, 84347141, 84347649, 84348173, 84476163, 84476175, 84477185,
    84477191, 84477701, 84477707, 84478211, 84479749, 84479755, 84607241, 84607747,
    84608261, 84608783, 84609281, 84609799, 84610817, 84738305, 84738309, 84738319,
    84739331, 84740875, 84741379, 84869387, 84869891, 84870413, 84870913, 84871431,
    84871937, 117506309, 117506819, 117506823, 117506827, 117506831, 117507333, 117507843,
    117507847, 117507851, 117508357, 117508361, 117508367, 117508867, 117509383, 117509891,
    117637379, 117637383, 117637387, 117637897, 117638403, 117638407, 117639425, 117640449,
    117640965, 117640973, 117768449, 117768965, 117769473, 117769989, 117769993, 117771009,
    117899523, 117900033, 117900041, 117900547, 117900551, 117900559, 117901057, 117901571,
    117901575, 117901583, 117902091, 117903111, 118030599, 118031107, 118031117, 118031621,
    118032131, 118033157, 118033665, 118033673, 118161667, 118162177, 118162181, 118162699,
    118163205, 118163721, 118164237, 118165255, 118293261, 118294787, 118423811, 118423815,
    118424833, 118424837, 118425355, 151060737, 151060745, 151061253, 151061761, 151061769,
    151061775, 151062277, 151062787, 151063297, 151064321, 151191813, 151191823, 151192323,
    151192327, 151192837, 151193345, 151193355, 151193863, 151194371, 151194379, 151322883,
    151322887, 151323393, 151323403, 151323907, 151324423, 151324929, 151325455, 151325957,
    151326465, 151453961, 151454467, 151454471, 151454977, 151454981, 151455491, 151455499,
    151585025, 151585029, 151586057, 151586575, 151587073, 151588611, 151716107, 151716111,
    151717123, 151719173, 151847687, 151848713, 151850241, 151978753, 151978763, 151979777,
    151980295, 151980803, 184615173, 184615681, 184615689, 184616197, 184617217, 184617225,
    184617231, 184617733, 184618253, 184618761, 184746243, 184746247, 184746251, 184746757,
    184747267, 184747781, 184749829, 184877313, 184877827, 184878343, 184878849, 184878861,
    184879879, 185008389, 185008399, 185008897, 185009423, 185010441, 185010947, 185011467,
    185011975, 185139459, 185139465, 185140481, 185140997, 185141517, 185271045, 185271565,
    185273091, 185273095, 185403653, 185532677, 185532681, 185533701, 218170115, 218170119,
    218170123, 218171139, 218171143, 218172673, 218300673, 218301697, 218301711, 218303753,
    218432261, 218433289, 218433797, 218434315, 218434821, 218435329, 218562817, 218563337,
    218563843, 218564865, 218694923, 218695943, 218696965, 218824961, 218824967, 218826505,
    218828033, 218956043, 218958081, 219087619, 219087623, 251724033, 251724041, 251724047,
    251725057, 251725061, 251725581, 251726081, 251726601, 251727109, 251855109, 251855619,
    251856137, 251857159, 251857163, 251986179, 251986185, 251986689, 251986701, 251987203,
    251987713, 251988739, 252117253, 252118789, 252118795, 252119815, 252248323, 252248331,
    252248839, 252249345, 252250881, 252380421, 252381445, 252510469, 252512003, 252641537
};

.visible .entry iq3_s_gemv_warp_reduce(
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
    @%p0 bra $L_s3_exit;

    mov.f32 %f0, 0f00000000;

    // nb = ceil(k_dim / 256)
    add.u32 %r4, %r3, 255;
    shr.u32 %r4, %r4, 8;

    // row_base = w_ptr + ctaid * nb * 110
    mul.lo.u32 %r5, %r4, 110;
    mul.wide.u32 %rd3, %r1, %r5;
    add.u64 %rd3, %rd1, %rd3;

    // ib = tid >> 2   (0..8),  l = tid & 3   (0..4)
    shr.u32 %r6, %r0, 2;
    and.b32 %r7, %r0, 3;

    mov.u64 %rd10, iq3s_grid_g;

    mov.u32 %r8, 0;                      // blk

$L_s3_blk:
    setp.ge.u32 %p1, %r8, %r4;
    @%p1 bra $L_s3_blk_end;

    mul.wide.u32 %rd4, %r8, 110;
    add.u64 %rd4, %rd3, %rd4;

    // d (f16 at +0)
    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;

    // sc = scales[ib >> 1]   (scales at +106)
    shr.u32 %r9, %r6, 1;
    cvt.u64.u32 %rd5, %r9;
    add.u64 %rd5, %rd4, %rd5;
    add.u64 %rd5, %rd5, 106;
    ld.global.u8 %r10, [%rd5];

    // nibble = ib & 1 ? (sc >> 4) : (sc & 0xf)
    and.b32 %r11, %r6, 1;
    shl.b32 %r12, %r11, 2;               // 0 or 4
    shr.u32 %r13, %r10, %r12;
    and.b32 %r13, %r13, 15;

    // db = d * (1 + 2*nib_val)
    cvt.rn.f32.u32 %f2, %r13;
    add.f32 %f2, %f2, %f2;               // 2*v
    add.f32 %f2, %f2, 0f3F800000;        // +1.0
    mul.f32 %f3, %f1, %f2;               // db

    // qh_byte = qh[ib]   (qh at +66)
    cvt.u64.u32 %rd6, %r6;
    add.u64 %rd6, %rd4, %rd6;
    add.u64 %rd6, %rd6, 66;
    ld.global.u8 %r14, [%rd6];

    // qs base = +2 + 8*ib + 2*l
    shl.b32 %r15, %r6, 3;
    shl.b32 %r16, %r7, 1;
    add.u32 %r15, %r15, %r16;
    cvt.u64.u32 %rd7, %r15;
    add.u64 %rd7, %rd4, %rd7;
    add.u64 %rd7, %rd7, 2;
    ld.global.u8 %r17, [%rd7];           // qs[8ib+2l]
    ld.global.u8 %r18, [%rd7+1];         // qs[8ib+2l+1]

    // high bits: bit(2l) and bit(2l+1) of qh_byte, each placed at 256
    shr.u32 %r19, %r14, %r16;            // qh >> 2l
    and.b32 %r20, %r19, 1;
    shl.b32 %r20, %r20, 8;
    or.b32 %r17, %r17, %r20;             // i1
    shr.u32 %r21, %r19, 1;               // qh >> (2l+1)
    and.b32 %r21, %r21, 1;
    shl.b32 %r21, %r21, 8;
    or.b32 %r18, %r18, %r21;             // i2

    // g1 = grid[i1], g2 = grid[i2]
    mul.wide.u32 %rd8, %r17, 4;
    add.u64 %rd8, %rd10, %rd8;
    ld.global.u32 %r22, [%rd8];
    mul.wide.u32 %rd9, %r18, 4;
    add.u64 %rd9, %rd10, %rd9;
    ld.global.u32 %r23, [%rd9];

    // sign_byte = signs[4*ib + l]   (signs at +74)
    shl.b32 %r24, %r6, 2;
    add.u32 %r24, %r24, %r7;
    cvt.u64.u32 %rd11, %r24;
    add.u64 %rd11, %rd4, %rd11;
    add.u64 %rd11, %rd11, 74;
    ld.global.u8 %r25, [%rd11];

    // col0 = blk*256 + 32*ib + 8*l
    shl.b32 %r26, %r8, 8;
    shl.b32 %r27, %r6, 5;
    add.u32 %r26, %r26, %r27;
    shl.b32 %r28, %r7, 3;
    add.u32 %r26, %r26, %r28;

    mov.u32 %r29, 0;                     // j = 0..4

$L_s3_j:
    setp.ge.u32 %p2, %r29, 4;
    @%p2 bra $L_s3_j_end;

    shl.b32 %r30, %r29, 3;               // 8*j
    shr.u32 %r31, %r22, %r30;
    and.b32 %r31, %r31, 255;             // m1
    shr.u32 %r32, %r23, %r30;
    and.b32 %r32, %r32, 255;             // m2

    // s1 = bit j of sign_byte, s2 = bit j+4
    shr.u32 %r33, %r25, %r29;
    and.b32 %r33, %r33, 1;
    add.u32 %r34, %r29, 4;
    shr.u32 %r35, %r25, %r34;
    and.b32 %r35, %r35, 1;

    cvt.rn.f32.u32 %f4, %r31;
    mul.f32 %f4, %f4, %f3;               // db * m1
    setp.ne.u32 %p3, %r33, 0;
    @%p3 neg.f32 %f4, %f4;

    cvt.rn.f32.u32 %f5, %r32;
    mul.f32 %f5, %f5, %f3;               // db * m2
    setp.ne.u32 %p4, %r35, 0;
    @%p4 neg.f32 %f5, %f5;

    // x[col0 + j]
    add.u32 %r36, %r26, %r29;
    setp.ge.u32 %p5, %r36, %r3;
    @%p5 bra $L_s3_skip1;
    mul.wide.u32 %rd12, %r36, 4;
    add.u64 %rd12, %rd2, %rd12;
    ld.global.f32 %f6, [%rd12];
    fma.rn.f32 %f0, %f4, %f6, %f0;
$L_s3_skip1:

    // x[col0 + j + 4]
    add.u32 %r37, %r36, 4;
    setp.ge.u32 %p6, %r37, %r3;
    @%p6 bra $L_s3_skip2;
    mul.wide.u32 %rd13, %r37, 4;
    add.u64 %rd13, %rd2, %rd13;
    ld.global.f32 %f7, [%rd13];
    fma.rn.f32 %f0, %f5, %f7, %f0;
$L_s3_skip2:

    add.u32 %r29, %r29, 1;
    bra $L_s3_j;

$L_s3_j_end:
    add.u32 %r8, %r8, 1;
    bra $L_s3_blk;

$L_s3_blk_end:
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

    setp.ne.u32 %p7, %r0, 0;
    @%p7 bra $L_s3_exit;

    mul.wide.u32 %rd14, %r1, 4;
    add.u64 %rd14, %rd0, %rd14;
    st.global.f32 [%rd14], %f0;

$L_s3_exit:
    ret;
}
",
    )
}

/// #3869: IQ4_NL (GGML type 20) GEMV, row-major.
///
/// The odd one out: IQ4_NL is a **32-element** block in 18 bytes
/// (`ggml-common.h`: `#define QK4_NL 32`, and `sizeof(block_iq4_nl) ==
/// sizeof(ggml_half) + QK4_NL/2` = 18), while every other IQ type here is a
/// 256-element super-block. There are no sub-block scales: one f16 `d` covers
/// the whole block.
///
/// It shares IQ4_XS's codebook exactly. IQ4_XS is this layout with a 6-bit
/// per-sub-block scale layered on (`dl = d * (ls - 32)`), so the nibble decode
/// below is what `iq4_xs_gemv_warp_reduce` does inside one of its eight
/// sub-blocks, with `dl` replaced by `d`.
///
/// Thread mapping: 32 threads, 32 elements per block, one element each. Element
/// `tid` takes byte `tid & 15`, low nibble when `tid < 16` and high nibble when
/// `tid >= 16`, which is the reference's `y[j]` / `y[j + QK4_NL/2]` split and
/// NOT adjacent nibbles. Reading them adjacently transposes each block's halves
/// and still produces plausible numbers.
///
/// LAYOUT-001: row-major. Row `ctaid` of an `[n, k]` weight starts at
/// `w_ptr + ctaid * ceil(k/32) * 18` and runs contiguously. No transpose and no
/// `*_colmajor` path.
///
/// NOTE FOR THE DISPATCH: `block_iq4_nl` is the identical C struct to
/// `block_q4_0`, so a tensor's SIZE can never tell them apart. The quant type
/// must come from the declared ggml type id, never from `from_size`.
fn generate_iq4_nl_gemv_ptx(k: u32, n: u32) -> String {
    let _ = (k, n);

    String::from(
        r"
.version 7.5
.target sm_70
.address_size 64

// The 16 non-linear IQ4_NL levels: quantize::iq_grids::KVALUES_IQ4NL.
.global .align 4 .s32 kvalues_iq4nl_b[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

.visible .entry iq4_nl_gemv_warp_reduce(
    .param .u64 y_ptr,
    .param .u64 w_ptr,
    .param .u64 x_ptr,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .reg .u32 %r<40>;
    .reg .u64 %rd<24>;
    .reg .f32 %f<20>;
    .reg .b16 %h<4>;
    .reg .pred %p<10>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;

    ld.param.u32 %r2, [n_dim];
    ld.param.u32 %r3, [k_dim];
    ld.param.u64 %rd0, [y_ptr];
    ld.param.u64 %rd1, [w_ptr];
    ld.param.u64 %rd2, [x_ptr];

    setp.ge.u32 %p0, %r1, %r2;
    @%p0 bra $L_nl_exit;

    mov.f32 %f0, 0f00000000;

    // nb = ceil(k_dim / 32)
    add.u32 %r4, %r3, 31;
    shr.u32 %r4, %r4, 5;

    // row_base = w_ptr + ctaid * nb * 18
    mul.lo.u32 %r5, %r4, 18;
    mul.wide.u32 %rd3, %r1, %r5;
    add.u64 %rd3, %rd1, %rd3;

    // per-thread: jlow = tid & 15 (byte index), jhalf = tid >> 4 (nibble select)
    and.b32 %r7, %r0, 15;
    shr.u32 %r6, %r0, 4;

    // nibble shift = jhalf * 4
    shl.b32 %r19, %r6, 2;

    // codebook base
    mov.u64 %rd10, kvalues_iq4nl_b;

    mov.u32 %r8, 0;

$L_nl_blk:
    setp.ge.u32 %p1, %r8, %r4;
    @%p1 bra $L_nl_blk_end;

    // blk_addr = row_base + blk * 18
    mul.wide.u32 %rd4, %r8, 18;
    add.u64 %rd4, %rd3, %rd4;

    // d (f16 at +0)
    ld.global.b16 %h0, [%rd4];
    cvt.f32.f16 %f1, %h0;

    // byte = qs[jlow], qs starts at +2
    cvt.u64.u32 %rd7, %r7;
    add.u64 %rd7, %rd4, %rd7;
    add.u64 %rd7, %rd7, 2;
    ld.global.u8 %r18, [%rd7];

    // nib = jhalf ? (byte >> 4) : (byte & 0xf)
    shr.u32 %r20, %r18, %r19;
    and.b32 %r20, %r20, 15;

    // w = d * kvalues_iq4nl[nib]
    mul.wide.u32 %rd8, %r20, 4;
    add.u64 %rd8, %rd10, %rd8;
    ld.global.s32 %r21, [%rd8];
    cvt.rn.f32.s32 %f4, %r21;
    mul.f32 %f5, %f1, %f4;

    // x_idx = blk*32 + tid
    shl.b32 %r22, %r8, 5;
    add.u32 %r22, %r22, %r0;

    setp.ge.u32 %p3, %r22, %r3;
    @%p3 bra $L_nl_skip;

    mul.wide.u32 %rd9, %r22, 4;
    add.u64 %rd9, %rd2, %rd9;
    ld.global.f32 %f6, [%rd9];
    fma.rn.f32 %f0, %f5, %f6, %f0;

$L_nl_skip:
    add.u32 %r8, %r8, 1;
    bra $L_nl_blk;

$L_nl_blk_end:
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
    @%p4 bra $L_nl_exit;

    mul.wide.u32 %rd11, %r1, 4;
    add.u64 %rd11, %rd0, %rd11;
    st.global.f32 [%rd11], %f0;

$L_nl_exit:
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
