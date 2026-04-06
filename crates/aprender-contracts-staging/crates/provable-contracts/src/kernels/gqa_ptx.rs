/// PTX assembly for Grouped Query Attention.
///
/// One block per query head. Shared memory holds KV cache for the mapped
/// KV head. Computes QK^T, softmax, and weighted V sum per head.
pub fn gqa_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry gqa_kernel(
    .param .u64 Q,
    .param .u64 K,
    .param .u64 V,
    .param .u64 OUT,
    .param .u32 SEQ_LEN,
    .param .u32 DK,
    .param .u32 DV,
    .param .u32 NUM_HEADS,
    .param .u32 NUM_KV_HEADS
) {
    .reg .u32 %tid, %bid, %seq_len, %dk, %dv;
    .reg .u32 %num_heads, %num_kv_heads, %heads_per_kv;
    .reg .u32 %head, %kv_head, %i, %j, %kk, %tmp32;
    .reg .u32 %q_off, %k_off, %v_off, %o_off;
    .reg .u32 %head_stride_q, %head_stride_k, %head_stride_v, %head_stride_o;
    .reg .u64 %q_ptr, %k_ptr, %v_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %dot, %q_val, %k_val, %v_val, %scale;
    .reg .f32 %score, %max_val, %sum, %exp_val, %weight, %acc;
    .reg .f32 %dk_f;
    .reg .pred %p_j, %p_k, %p_d, %p_seq;
    .shared .f32 scores[256];
    .shared .f32 kv_cache[512];

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    // Load params
    ld.param.u32 %seq_len, [SEQ_LEN];
    ld.param.u32 %dk, [DK];
    ld.param.u32 %dv, [DV];
    ld.param.u32 %num_heads, [NUM_HEADS];
    ld.param.u32 %num_kv_heads, [NUM_KV_HEADS];
    ld.param.u64 %q_ptr, [Q];
    ld.param.u64 %k_ptr, [K];
    ld.param.u64 %v_ptr, [V];
    ld.param.u64 %out_ptr, [OUT];

    // bid = query head index
    mov.u32 %head, %bid;

    // heads_per_kv = num_heads / num_kv_heads
    div.u32 %heads_per_kv, %num_heads, %num_kv_heads;

    // kv_head = head / heads_per_kv
    div.u32 %kv_head, %head, %heads_per_kv;

    // Compute strides
    mul.lo.u32 %head_stride_q, %seq_len, %dk;
    mul.lo.u32 %head_stride_k, %seq_len, %dk;
    mul.lo.u32 %head_stride_v, %seq_len, %dv;
    mul.lo.u32 %head_stride_o, %seq_len, %dv;

    // Scale = 1/sqrt(dk)
    cvt.rn.f32.u32 %dk_f, %dk;
    sqrt.approx.f32 %scale, %dk_f;
    rcp.approx.f32 %scale, %scale;

    // For simplicity, process one query position per thread (tid = query pos)
    // For each query position tid, compute full attention row
    setp.ge.u32 %p_seq, %tid, %seq_len;
    @%p_seq bra EXIT;

    // Compute Q offset for this head and position
    mul.lo.u32 %q_off, %head, %head_stride_q;
    mad.lo.u32 %q_off, %tid, %dk, %q_off;

    // Compute K, V offsets for kv_head
    mul.lo.u32 %k_off, %kv_head, %head_stride_k;
    mul.lo.u32 %v_off, %kv_head, %head_stride_v;

    // Phase 1: Compute scores[j] = Q[tid] . K[j] / sqrt(dk) for j in 0..seq_len
    mov.u32 %j, 0;
SCORE_LOOP:
    setp.ge.u32 %p_j, %j, %seq_len;
    @%p_j bra SCORE_DONE;

    mov.f32 %dot, 0f00000000;
    mov.u32 %kk, 0;
DOT_LOOP:
    setp.ge.u32 %p_k, %kk, %dk;
    @%p_k bra DOT_DONE;

    // Q[head][tid][kk]
    add.u32 %tmp32, %q_off, %kk;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %q_ptr, %off64;
    ld.global.f32 %q_val, [%addr];

    // K[kv_head][j][kk]
    mad.lo.u32 %tmp32, %j, %dk, %kk;
    add.u32 %tmp32, %tmp32, %k_off;
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

    add.u32 %j, %j, 1;
    bra SCORE_LOOP;
SCORE_DONE:

    bar.sync 0;

    // Phase 2: Softmax over scores[0..seq_len]
    mov.f32 %max_val, 0fFF7FFFFF;
    mov.u32 %j, 0;
MAX_LOOP:
    setp.ge.u32 %p_j, %j, %seq_len;
    @%p_j bra MAX_DONE;
    mul.wide.u32 %off64, %j, 4;
    add.u64 %addr, %off64, scores;
    ld.shared.f32 %score, [%addr];
    max.f32 %max_val, %max_val, %score;
    add.u32 %j, %j, 1;
    bra MAX_LOOP;
MAX_DONE:

    mov.f32 %sum, 0f00000000;
    mov.u32 %j, 0;
EXP_LOOP:
    setp.ge.u32 %p_j, %j, %seq_len;
    @%p_j bra EXP_DONE;
    mul.wide.u32 %off64, %j, 4;
    add.u64 %addr, %off64, scores;
    ld.shared.f32 %score, [%addr];
    sub.f32 %score, %score, %max_val;
    mul.f32 %score, %score, 0f3FB8AA3B;
    ex2.approx.f32 %exp_val, %score;
    st.shared.f32 [%addr], %exp_val;
    add.f32 %sum, %sum, %exp_val;
    add.u32 %j, %j, 1;
    bra EXP_LOOP;
EXP_DONE:

    rcp.approx.f32 %sum, %sum;
    mov.u32 %j, 0;
NORM_LOOP:
    setp.ge.u32 %p_j, %j, %seq_len;
    @%p_j bra NORM_DONE;
    mul.wide.u32 %off64, %j, 4;
    add.u64 %addr, %off64, scores;
    ld.shared.f32 %score, [%addr];
    mul.f32 %score, %score, %sum;
    st.shared.f32 [%addr], %score;
    add.u32 %j, %j, 1;
    bra NORM_LOOP;
NORM_DONE:

    // Phase 3: output[head][tid][d] = sum_j scores[j] * V[kv_head][j][d]
    mul.lo.u32 %o_off, %head, %head_stride_o;
    mad.lo.u32 %o_off, %tid, %dv, %o_off;

    mov.u32 %i, 0;
OUT_LOOP:
    setp.ge.u32 %p_d, %i, %dv;
    @%p_d bra OUT_DONE;

    mov.f32 %acc, 0f00000000;
    mov.u32 %j, 0;
WEIGHT_LOOP:
    setp.ge.u32 %p_j, %j, %seq_len;
    @%p_j bra WEIGHT_DONE;

    mul.wide.u32 %off64, %j, 4;
    add.u64 %addr, %off64, scores;
    ld.shared.f32 %weight, [%addr];

    // V[kv_head][j][i]
    mad.lo.u32 %tmp32, %j, %dv, %i;
    add.u32 %tmp32, %tmp32, %v_off;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %v_ptr, %off64;
    ld.global.f32 %v_val, [%addr];

    fma.rn.f32 %acc, %weight, %v_val, %acc;
    add.u32 %j, %j, 1;
    bra WEIGHT_LOOP;
WEIGHT_DONE:

    // Store output
    add.u32 %tmp32, %o_off, %i;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %out_ptr, %off64;
    st.global.f32 [%addr], %acc;

    add.u32 %i, %i, 1;
    bra OUT_LOOP;
OUT_DONE:

EXIT:
    ret;
}
"#
}
