
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
