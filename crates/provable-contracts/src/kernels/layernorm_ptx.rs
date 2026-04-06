/// PTX assembly for LayerNorm kernel.
///
/// Two-pass reduction (sum for mean, then sum-of-squares for variance),
/// followed by normalize + affine transform. 1 block per vector, 256 threads.
pub fn layernorm_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// LayerNorm kernel: 1 block per vector, 256 threads per block.
// Three-pass: sum for mean, sum-of-squares for variance, normalize + affine.
.visible .entry layernorm_kernel(
    .param .u64 input_ptr,
    .param .u64 gamma_ptr,
    .param .u64 beta_ptr,
    .param .u64 output_ptr,
    .param .u32 n,
    .param .f32 eps
)
{
    .reg .u32 %tid, %n, %i, %lane, %warp_id, %mask;
    .reg .u64 %in_base, %g_base, %b_base, %out_base, %addr;
    .reg .f32 %val, %diff, %sq;
    .reg .f32 %sum_local, %sum_warp, %mean;
    .reg .f32 %var_local, %var_warp, %variance, %inv_std;
    .reg .f32 %eps, %nf, %gamma_val, %beta_val, %normed, %result;
    .reg .pred %p;
    .shared .f32 smem[32];

    ld.param.u64 %in_base, [input_ptr];
    ld.param.u64 %g_base, [gamma_ptr];
    ld.param.u64 %b_base, [beta_ptr];
    ld.param.u64 %out_base, [output_ptr];
    ld.param.u32 %n, [n];
    ld.param.f32 %eps, [eps];

    mov.u32 %tid, %tid.x;
    mov.u32 %mask, 0xFFFFFFFF;

    // --- Pass 1: compute sum for mean ---
    mov.f32 %sum_local, 0f00000000;
    mov.u32 %i, %tid;
sum_loop:
    setp.ge.u32 %p, %i, %n;
    @%p bra sum_done;
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    add.f32 %sum_local, %sum_local, %val;
    add.u32 %i, %i, 256;
    bra sum_loop;
sum_done:

    // Warp-level sum reduction
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

    setp.lt.u32 %p, %tid, 8;
    @%p ld.shared.f32 %sum_local, [smem + %tid * 4];
    @!%p mov.f32 %sum_local, 0f00000000;
    shfl.sync.down.b32 %sum_warp, %sum_local, 4, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 2, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;
    shfl.sync.down.b32 %sum_warp, %sum_local, 1, 31, %mask;
    add.f32 %sum_local, %sum_local, %sum_warp;

    // Compute mean = sum / n
    setp.eq.u32 %p, %tid, 0;
    cvt.rn.f32.u32 %nf, %n;
    div.approx.f32 %mean, %sum_local, %nf;
    @%p st.shared.f32 [smem], %mean;
    bar.sync 0;
    ld.shared.f32 %mean, [smem];

    // --- Pass 2: compute variance ---
    mov.f32 %var_local, 0f00000000;
    mov.u32 %i, %tid;
var_loop:
    setp.ge.u32 %p, %i, %n;
    @%p bra var_done;
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    sub.f32 %diff, %val, %mean;
    mul.f32 %sq, %diff, %diff;
    add.f32 %var_local, %var_local, %sq;
    add.u32 %i, %i, 256;
    bra var_loop;
var_done:

    // Warp-level variance reduction
    shfl.sync.down.b32 %var_warp, %var_local, 16, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 8, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 4, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 2, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 1, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;

    and.b32 %lane, %tid, 31;
    shr.b32 %warp_id, %tid, 5;
    setp.eq.u32 %p, %lane, 0;
    @%p st.shared.f32 [smem + %warp_id * 4], %var_local;
    bar.sync 0;

    setp.lt.u32 %p, %tid, 8;
    @%p ld.shared.f32 %var_local, [smem + %tid * 4];
    @!%p mov.f32 %var_local, 0f00000000;
    shfl.sync.down.b32 %var_warp, %var_local, 4, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 2, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;
    shfl.sync.down.b32 %var_warp, %var_local, 1, 31, %mask;
    add.f32 %var_local, %var_local, %var_warp;

    // Compute inv_std = rsqrt(variance/n + eps)
    setp.eq.u32 %p, %tid, 0;
    div.approx.f32 %variance, %var_local, %nf;
    add.f32 %variance, %variance, %eps;
    rsqrt.approx.f32 %inv_std, %variance;
    @%p st.shared.f32 [smem], %inv_std;
    bar.sync 0;
    ld.shared.f32 %inv_std, [smem];

    // --- Pass 3: normalize + affine transform ---
    mov.u32 %i, %tid;
norm_loop:
    setp.ge.u32 %p, %i, %n;
    @%p bra norm_done;
    // Load input[i]
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    // Load gamma[i]
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %g_base, %addr;
    ld.global.f32 %gamma_val, [%addr];
    // Load beta[i]
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %b_base, %addr;
    ld.global.f32 %beta_val, [%addr];
    // output[i] = gamma * (x - mean) * inv_std + beta
    sub.f32 %diff, %val, %mean;
    mul.f32 %normed, %diff, %inv_std;
    fma.rn.f32 %result, %gamma_val, %normed, %beta_val;
    // Store output[i]
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out_base, %addr;
    st.global.f32 [%addr], %result;
    add.u32 %i, %i, 256;
    bra norm_loop;
norm_done:

    bar.sync 0;
    ret;
}
"#
}
