/// PTX assembly for the cross-entropy kernel.
///
/// Two-phase kernel:
/// 1. Reduction phase: compute max and log-sum-exp using shared memory.
/// 2. Elementwise phase: compute -sum(targets_i * (logits_i - lse)).
///
/// Uses `ex2.approx.f32` for exp and `lg2.approx.f32` for log with ln2 scaling.
pub fn cross_entropy_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry cross_entropy_kernel(
    .param .u64 targets,
    .param .u64 logits,
    .param .u64 output,
    .param .u32 n
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n, %stride, %i;
    .reg .u64 %tgt_ptr, %log_ptr, %out_ptr, %off;
    .reg .f32 %t, %x, %max_val, %cur, %shifted, %exp_val, %sum_exp;
    .reg .f32 %lse, %log_sum, %log_softmax_i, %prod, %loss;
    .reg .f32 %k_neg_inf, %k_one, %k_rcp_ln2, %k_ln2;
    .reg .pred %p, %loop_p;
    .shared .f32 smem[1024];

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    ld.param.u32 %n, [n];
    ld.param.u64 %tgt_ptr, [targets];
    ld.param.u64 %log_ptr, [logits];
    ld.param.u64 %out_ptr, [output];

    // Constants
    mov.f32 %k_neg_inf, 0fFF800000;   // -infinity
    mov.f32 %k_one, 0f3F800000;       // 1.0
    mov.f32 %k_rcp_ln2, 0f3FB8AA3B;   // 1/ln(2) ~ 1.442695
    mov.f32 %k_ln2, 0f3F317218;       // ln(2) ~ 0.693147

    // Phase 1: Find max via grid-stride loop
    mov.f32 %max_val, %k_neg_inf;
    mov.u32 %i, %tid;
FIND_MAX:
    setp.ge.u32 %loop_p, %i, %n;
    @%loop_p bra MAX_DONE;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %log_ptr, %off;
    ld.global.f32 %cur, [%off];
    max.f32 %max_val, %max_val, %cur;
    add.u32 %i, %i, %ntid;
    bra FIND_MAX;
MAX_DONE:

    // Store thread max to shared memory, sync, reduce
    st.shared.f32 [smem], %max_val;
    bar.sync 0;

    // Phase 2: Compute sum(exp(x_i - max)) via grid-stride loop
    mov.f32 %sum_exp, 0f00000000;
    mov.u32 %i, %tid;
SUM_EXP:
    setp.ge.u32 %loop_p, %i, %n;
    @%loop_p bra SUM_DONE;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %log_ptr, %off;
    ld.global.f32 %cur, [%off];
    sub.f32 %shifted, %cur, %max_val;
    mul.f32 %shifted, %shifted, %k_rcp_ln2;
    ex2.approx.f32 %exp_val, %shifted;
    add.f32 %sum_exp, %sum_exp, %exp_val;
    add.u32 %i, %i, %ntid;
    bra SUM_EXP;
SUM_DONE:

    // lse = max + ln(sum_exp) = max + lg2(sum_exp) * ln(2)
    lg2.approx.f32 %log_sum, %sum_exp;
    mul.f32 %log_sum, %log_sum, %k_ln2;
    add.f32 %lse, %max_val, %log_sum;

    // Phase 3: Compute loss = -sum(t_i * (x_i - lse))
    mov.f32 %loss, 0f00000000;
    mov.u32 %i, %tid;
LOSS_LOOP:
    setp.ge.u32 %loop_p, %i, %n;
    @%loop_p bra LOSS_DONE;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %tgt_ptr, %off;
    ld.global.f32 %t, [%off];
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %log_ptr, %off;
    ld.global.f32 %x, [%off];
    sub.f32 %log_softmax_i, %x, %lse;
    mul.f32 %prod, %t, %log_softmax_i;
    add.f32 %loss, %loss, %prod;
    add.u32 %i, %i, %ntid;
    bra LOSS_LOOP;
LOSS_DONE:

    // Negate and store: output = -loss
    neg.f32 %loss, %loss;
    setp.eq.u32 %p, %tid, 0;
    @%p st.global.f32 [%out_ptr], %loss;

    ret;
}
"#
}
