/// PTX assembly for softmax kernel.
///
/// Reduction kernel with shared memory. 1 block per row, 256 threads.
/// Uses warp shuffle `shfl.sync.down.b32` for intra-warp reduction and
/// `.shared .f32 smem[32]` for cross-warp communication.
/// Multi-pass: max -> exp+sum -> normalize.
pub fn softmax_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// Softmax kernel: 1 block per row, 256 threads per block.
// Three-pass reduction: max, exp+sum, normalize.
.visible .entry softmax_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 n
)
{
    .reg .u32 %tid, %n, %i, %lane, %warp_id, %mask;
    .reg .u64 %in_base, %out_base, %addr;
    .reg .f32 %val, %max_local, %max_warp, %max_global;
    .reg .f32 %exp_val, %sum_local, %sum_warp, %sum_global, %inv_sum;
    .reg .pred %p;
    .shared .f32 smem[32];

    ld.param.u64 %in_base, [input_ptr];
    ld.param.u64 %out_base, [output_ptr];
    ld.param.u32 %n, [n];

    mov.u32 %tid, %tid.x;
    mov.u32 %mask, 0xFFFFFFFF;

    // --- Pass 1: find max ---
    mov.f32 %max_local, 0fFF800000;  // -infinity
    mov.u32 %i, %tid;
pass1_loop:
    setp.ge.u32 %p, %i, %n;
    @%p bra pass1_done;
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    max.f32 %max_local, %max_local, %val;
    add.u32 %i, %i, 256;
    bra pass1_loop;
pass1_done:

    // Warp-level max reduction via shuffle
    shfl.sync.down.b32 %max_warp, %max_local, 16, 31, %mask;
    max.f32 %max_local, %max_local, %max_warp;
    shfl.sync.down.b32 %max_warp, %max_local, 8, 31, %mask;
    max.f32 %max_local, %max_local, %max_warp;
    shfl.sync.down.b32 %max_warp, %max_local, 4, 31, %mask;
    max.f32 %max_local, %max_local, %max_warp;
    shfl.sync.down.b32 %max_warp, %max_local, 2, 31, %mask;
    max.f32 %max_local, %max_local, %max_warp;
    shfl.sync.down.b32 %max_warp, %max_local, 1, 31, %mask;
    max.f32 %max_local, %max_local, %max_warp;

    // Write warp max to shared memory
    and.b32 %lane, %tid, 31;
    shr.b32 %warp_id, %tid, 5;
    setp.eq.u32 %p, %lane, 0;
    @%p st.shared.f32 [smem + %warp_id * 4], %max_local;

    bar.sync 0;

    // First warp reduces across warps (8 warps for 256 threads)
    setp.lt.u32 %p, %tid, 8;
    @%p ld.shared.f32 %max_local, [smem + %tid * 4];
    @!%p mov.f32 %max_local, 0fFF800000;
    shfl.sync.down.b32 %max_warp, %max_local, 4, 31, %mask;
    max.f32 %max_local, %max_local, %max_warp;
    shfl.sync.down.b32 %max_warp, %max_local, 2, 31, %mask;
    max.f32 %max_local, %max_local, %max_warp;
    shfl.sync.down.b32 %max_warp, %max_local, 1, 31, %mask;
    max.f32 %max_local, %max_local, %max_warp;

    // Broadcast global max via shared memory
    setp.eq.u32 %p, %tid, 0;
    @%p st.shared.f32 [smem], %max_local;
    bar.sync 0;
    ld.shared.f32 %max_global, [smem];

    // --- Pass 2: exp(x - max) and sum ---
    mov.f32 %sum_local, 0f00000000;  // 0.0
    mov.u32 %i, %tid;
pass2_loop:
    setp.ge.u32 %p, %i, %n;
    @%p bra pass2_done;
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    sub.f32 %val, %val, %max_global;
    // exp(val) via exp2(val * log2(e))
    mul.f32 %val, %val, 0f3FB8AA3B;  // log2(e) = 1.4426950408
    ex2.approx.f32 %exp_val, %val;
    // Store exp result to output
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out_base, %addr;
    st.global.f32 [%addr], %exp_val;
    add.f32 %sum_local, %sum_local, %exp_val;
    add.u32 %i, %i, 256;
    bra pass2_loop;
pass2_done:

    // Warp-level sum reduction via shuffle
    shfl.sync.down.b32 %sum_warp, %sum_local, 16, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 8, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 4, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 2, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 1, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;

    and.b32 %lane, %tid, 31;
    shr.b32 %warp_id, %tid, 5;
    setp.eq.u32 %p, %lane, 0;
    @%p st.shared.f32 [smem + %warp_id * 4], %sum_local;

    bar.sync 0;

    // First warp reduces across warps
    setp.lt.u32 %p, %tid, 8;
    @%p ld.shared.f32 %sum_local, [smem + %tid * 4];
    @!%p mov.f32 %sum_local, 0f00000000;
    shfl.sync.down.b32 %sum_warp, %sum_local, 4, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 2, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 1, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;

    // Broadcast inverse sum
    setp.eq.u32 %p, %tid, 0;
    @%p st.shared.f32 [smem], %sum_local;
    bar.sync 0;
    ld.shared.f32 %sum_global, [smem];
    rcp.approx.f32 %inv_sum, %sum_global;

    // --- Pass 3: normalize ---
    mov.u32 %i, %tid;
pass3_loop:
    setp.ge.u32 %p, %i, %n;
    @%p bra pass3_done;
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out_base, %addr;
    ld.global.f32 %val, [%addr];
    mul.f32 %val, %val, %inv_sum;
    st.global.f32 [%addr], %val;
    add.u32 %i, %i, 256;
    bra pass3_loop;
pass3_done:

    bar.sync 0;
    ret;
}
"#
}
