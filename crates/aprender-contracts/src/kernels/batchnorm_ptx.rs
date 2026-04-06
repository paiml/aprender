/// PTX assembly for BatchNorm kernel (training mode).
///
/// One block per channel. Each block reduces across the batch dimension
/// to compute per-channel mean and variance, then normalizes.
pub fn batchnorm_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// BatchNorm kernel (training): 1 block per channel, 256 threads per block.
// Each block computes mean/var for its channel, normalizes, and updates running stats.
// Input layout: [N, C] row-major, element (n, ch) = input[n * C + ch].
.visible .entry batchnorm_kernel(
    .param .u64 input_ptr,
    .param .u64 gamma_ptr,
    .param .u64 beta_ptr,
    .param .u64 output_ptr,
    .param .u64 running_mean_ptr,
    .param .u64 running_var_ptr,
    .param .u32 batch_size,
    .param .u32 channels,
    .param .f32 eps,
    .param .f32 momentum
)
{
    .reg .u32 %tid, %ch, %n_batch, %n_ch, %i, %idx, %stride;
    .reg .u32 %lane, %warp_id, %mask;
    .reg .u64 %in_base, %g_base, %b_base, %out_base;
    .reg .u64 %rm_base, %rv_base, %addr;
    .reg .f32 %val, %diff, %sq;
    .reg .f32 %sum_local, %sum_warp, %batch_mean;
    .reg .f32 %var_local, %var_warp, %batch_var;
    .reg .f32 %inv_std, %eps, %momentum, %nf;
    .reg .f32 %gamma_val, %beta_val, %normed, %result;
    .reg .f32 %old_rm, %old_rv, %new_rm, %new_rv, %one_minus_m;
    .reg .pred %p;
    .shared .f32 smem[32];

    ld.param.u64 %in_base, [input_ptr];
    ld.param.u64 %g_base, [gamma_ptr];
    ld.param.u64 %b_base, [beta_ptr];
    ld.param.u64 %out_base, [output_ptr];
    ld.param.u64 %rm_base, [running_mean_ptr];
    ld.param.u64 %rv_base, [running_var_ptr];
    ld.param.u32 %n_batch, [batch_size];
    ld.param.u32 %n_ch, [channels];
    ld.param.f32 %eps, [eps];
    ld.param.f32 %momentum, [momentum];

    mov.u32 %tid, %tid.x;
    mov.u32 %ch, %ctaid.x;  // 1 block per channel
    mov.u32 %mask, 0xFFFFFFFF;

    // --- Pass 1: compute sum for mean ---
    mov.f32 %sum_local, 0f00000000;
    mov.u32 %i, %tid;
mean_loop:
    setp.ge.u32 %p, %i, %n_batch;
    @%p bra mean_done;
    // idx = i * channels + ch
    mad.lo.u32 %idx, %i, %n_ch, %ch;
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    add.f32 %sum_local, %sum_local, %val;
    add.u32 %i, %i, 256;
    bra mean_loop;
mean_done:

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

    // mean = sum / N
    setp.eq.u32 %p, %tid, 0;
    cvt.rn.f32.u32 %nf, %n_batch;
    div.approx.f32 %batch_mean, %sum_local, %nf;
    @%p st.shared.f32 [smem], %batch_mean;
    bar.sync 0;
    ld.shared.f32 %batch_mean, [smem];

    // --- Pass 2: compute variance ---
    mov.f32 %var_local, 0f00000000;
    mov.u32 %i, %tid;
var_loop:
    setp.ge.u32 %p, %i, %n_batch;
    @%p bra var_done;
    mad.lo.u32 %idx, %i, %n_ch, %ch;
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    sub.f32 %diff, %val, %batch_mean;
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

    // var = var_sum / N, inv_std = rsqrt(var + eps)
    setp.eq.u32 %p, %tid, 0;
    div.approx.f32 %batch_var, %var_local, %nf;
    add.f32 %batch_var, %batch_var, %eps;
    rsqrt.approx.f32 %inv_std, %batch_var;
    @%p st.shared.f32 [smem], %inv_std;

    // Also update running stats (thread 0 only)
    @%p {
        // running_mean = (1-m)*running_mean + m*batch_mean
        cvt.u64.u32 %addr, %ch;
        shl.b64 %addr, %addr, 2;
        add.u64 %addr, %rm_base, %addr;
        ld.global.f32 %old_rm, [%addr];
        mov.f32 %one_minus_m, 0f3F800000;
        sub.f32 %one_minus_m, %one_minus_m, %momentum;
        mul.f32 %new_rm, %one_minus_m, %old_rm;
        fma.rn.f32 %new_rm, %momentum, %batch_mean, %new_rm;
        st.global.f32 [%addr], %new_rm;

        // running_var = (1-m)*running_var + m*batch_var (before eps was added)
        // Recompute batch_var without eps
        sub.f32 %batch_var, %batch_var, %eps;
        cvt.u64.u32 %addr, %ch;
        shl.b64 %addr, %addr, 2;
        add.u64 %addr, %rv_base, %addr;
        ld.global.f32 %old_rv, [%addr];
        mul.f32 %new_rv, %one_minus_m, %old_rv;
        fma.rn.f32 %new_rv, %momentum, %batch_var, %new_rv;
        st.global.f32 [%addr], %new_rv;
    }

    bar.sync 0;
    ld.shared.f32 %inv_std, [smem];

    // Load gamma and beta for this channel
    cvt.u64.u32 %addr, %ch;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %g_base, %addr;
    ld.global.f32 %gamma_val, [%addr];
    cvt.u64.u32 %addr, %ch;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %b_base, %addr;
    ld.global.f32 %beta_val, [%addr];

    // --- Pass 3: normalize + affine ---
    mov.u32 %i, %tid;
norm_loop:
    setp.ge.u32 %p, %i, %n_batch;
    @%p bra norm_done;
    mad.lo.u32 %idx, %i, %n_ch, %ch;
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %val, [%addr];
    sub.f32 %diff, %val, %batch_mean;
    mul.f32 %normed, %diff, %inv_std;
    fma.rn.f32 %result, %gamma_val, %normed, %beta_val;
    cvt.u64.u32 %addr, %idx;
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
