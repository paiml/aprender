/// PTX assembly for fused scaled dot-product attention.
///
/// One block per query row. Each block computes the QK^T row, applies softmax
/// in shared memory, then computes the weighted sum over V.
pub fn attention_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry attention_kernel(
    .param .u64 Q,
    .param .u64 K,
    .param .u64 V,
    .param .u64 OUT,
    .param .u32 N,
    .param .u32 M,
    .param .u32 DK,
    .param .u32 DV
) {
    .reg .u32 %tid, %bid, %n, %m, %dk, %dv;
    .reg .u32 %i, %j, %kk, %tmp32;
    .reg .u64 %q_ptr, %k_ptr, %v_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %dot, %q_val, %k_val, %v_val, %scale, %score;
    .reg .f32 %max_val, %sum, %exp_val, %weight, %acc;
    .reg .f32 %dk_f, %correction;
    .reg .pred %p_j, %p_k, %p_d;
    .shared .f32 scores[1024];

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    // Load params
    ld.param.u32 %n, [N];
    ld.param.u32 %m, [M];
    ld.param.u32 %dk, [DK];
    ld.param.u32 %dv, [DV];
    ld.param.u64 %q_ptr, [Q];
    ld.param.u64 %k_ptr, [K];
    ld.param.u64 %v_ptr, [V];
    ld.param.u64 %out_ptr, [OUT];

    // bid = query row index (i)
    // tid = thread within block, used to parallelize over M and DV
    // Scale factor = 1/sqrt(dk)
    cvt.rn.f32.u32 %dk_f, %dk;
    sqrt.approx.f32 %scale, %dk_f;
    rcp.approx.f32 %scale, %scale;

    // Phase 1: Compute scores[j] = Q[bid] . K[j] / sqrt(dk)
    // Each thread handles one key index j = tid
    mov.u32 %j, %tid;
SCORE_LOOP:
    setp.ge.u32 %p_j, %j, %m;
    @%p_j bra SCORE_DONE;

    mov.f32 %dot, 0f00000000;
    mov.u32 %kk, 0;
DOT_LOOP:
    setp.ge.u32 %p_k, %kk, %dk;
    @%p_k bra DOT_DONE;

    // Load Q[bid * dk + kk]
    mad.lo.u32 %tmp32, %bid, %dk, %kk;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %q_ptr, %off64;
    ld.global.f32 %q_val, [%addr];

    // Load K[j * dk + kk]
    mad.lo.u32 %tmp32, %j, %dk, %kk;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %k_ptr, %off64;
    ld.global.f32 %k_val, [%addr];

    fma.rn.f32 %dot, %q_val, %k_val, %dot;
    add.u32 %kk, %kk, 1;
    bra DOT_LOOP;
DOT_DONE:

    mul.f32 %score, %dot, %scale;
    mul.wide.u32 %off64, %j, 4;
    add.u64 %addr, %off64, scores;
    st.shared.f32 [%addr], %score;

    add.u32 %j, %j, %ntid.x;
    bra SCORE_LOOP;
SCORE_DONE:

    bar.sync 0;

    // Phase 2: Softmax over scores[0..m] (thread 0 does serial softmax)
    setp.ne.u32 %p_j, %tid, 0;
    @%p_j bra SOFTMAX_DONE;

    // Find max
    mov.f32 %max_val, 0fFF7FFFFF;
    mov.u32 %j, 0;
MAX_LOOP:
    setp.ge.u32 %p_j, %j, %m;
    @%p_j bra MAX_DONE;
    mul.wide.u32 %off64, %j, 4;
    add.u64 %addr, %off64, scores;
    ld.shared.f32 %score, [%addr];
    max.f32 %max_val, %max_val, %score;
    add.u32 %j, %j, 1;
    bra MAX_LOOP;
MAX_DONE:

    // Exp and sum
    mov.f32 %sum, 0f00000000;
    mov.u32 %j, 0;
EXP_LOOP:
    setp.ge.u32 %p_j, %j, %m;
    @%p_j bra EXP_DONE;
    mul.wide.u32 %off64, %j, 4;
    add.u64 %addr, %off64, scores;
    ld.shared.f32 %score, [%addr];
    sub.f32 %score, %score, %max_val;
    // exp(x) = 2^(x / ln2)
    mul.f32 %score, %score, 0f3FB8AA3B;
    ex2.approx.f32 %exp_val, %score;
    st.shared.f32 [%addr], %exp_val;
    add.f32 %sum, %sum, %exp_val;
    add.u32 %j, %j, 1;
    bra EXP_LOOP;
EXP_DONE:

    // Normalize
    rcp.approx.f32 %sum, %sum;
    mov.u32 %j, 0;
NORM_LOOP:
    setp.ge.u32 %p_j, %j, %m;
    @%p_j bra NORM_DONE;
    mul.wide.u32 %off64, %j, 4;
    add.u64 %addr, %off64, scores;
    ld.shared.f32 %score, [%addr];
    mul.f32 %score, %score, %sum;
    st.shared.f32 [%addr], %score;
    add.u32 %j, %j, 1;
    bra NORM_LOOP;
NORM_DONE:

SOFTMAX_DONE:
    bar.sync 0;

    // Phase 3: output[bid][d] = sum_j scores[j] * V[j][d]
    // Each thread handles one output dimension d = tid
    mov.u32 %i, %tid;
OUT_LOOP:
    setp.ge.u32 %p_d, %i, %dv;
    @%p_d bra OUT_DONE;

    mov.f32 %acc, 0f00000000;
    mov.u32 %j, 0;
WEIGHT_LOOP:
    setp.ge.u32 %p_j, %j, %m;
    @%p_j bra WEIGHT_DONE;

    mul.wide.u32 %off64, %j, 4;
    add.u64 %addr, %off64, scores;
    ld.shared.f32 %weight, [%addr];

    // V[j * dv + i]
    mad.lo.u32 %tmp32, %j, %dv, %i;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %v_ptr, %off64;
    ld.global.f32 %v_val, [%addr];

    fma.rn.f32 %acc, %weight, %v_val, %acc;
    add.u32 %j, %j, 1;
    bra WEIGHT_LOOP;
WEIGHT_DONE:

    // Store output[bid * dv + i]
    mad.lo.u32 %tmp32, %bid, %dv, %i;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %out_ptr, %off64;
    st.global.f32 [%addr], %acc;

    add.u32 %i, %i, %ntid.x;
    bra OUT_LOOP;
OUT_DONE:

    ret;
}
"#
}
