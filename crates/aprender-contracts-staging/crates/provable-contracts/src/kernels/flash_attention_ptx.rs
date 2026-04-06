/// PTX assembly for Flash Attention.
///
/// One block per query row. Tiled loop over KV with shared memory, online
/// softmax maintaining running max and sum across tiles.
pub fn flash_attention_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry flash_attention_kernel(
    .param .u64 Q,
    .param .u64 K,
    .param .u64 V,
    .param .u64 OUT,
    .param .u32 N,
    .param .u32 D,
    .param .u32 TILE_SIZE
) {
    .reg .u32 %tid, %bid, %n, %d, %tile_size;
    .reg .u32 %i, %j, %kk, %dd, %tile_start, %tile_end, %tile_len;
    .reg .u32 %tmp32;
    .reg .u64 %q_ptr, %k_ptr, %v_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %dot, %q_val, %k_val, %v_val, %scale;
    .reg .f32 %score, %tile_max, %new_max, %running_max, %running_sum;
    .reg .f32 %correction, %weight, %acc, %d_f, %rcp_sum;
    .reg .pred %p_tile, %p_j, %p_k, %p_d, %p_end;
    .shared .f32 tile_scores[256];
    .shared .f32 tile_v[4096];

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    // Load params
    ld.param.u32 %n, [N];
    ld.param.u32 %d, [D];
    ld.param.u32 %tile_size, [TILE_SIZE];
    ld.param.u64 %q_ptr, [Q];
    ld.param.u64 %k_ptr, [K];
    ld.param.u64 %v_ptr, [V];
    ld.param.u64 %out_ptr, [OUT];

    // bid = query row index (i)
    // tid = thread within block (used to parallelize over D)

    // Scale = 1/sqrt(d)
    cvt.rn.f32.u32 %d_f, %d;
    sqrt.approx.f32 %scale, %d_f;
    rcp.approx.f32 %scale, %scale;

    // Online softmax state
    mov.f32 %running_max, 0fFF7FFFFF;
    mov.f32 %running_sum, 0f00000000;
    mov.f32 %acc, 0f00000000;

    // Tile loop
    mov.u32 %tile_start, 0;
TILE_LOOP:
    setp.ge.u32 %p_tile, %tile_start, %n;
    @%p_tile bra TILE_DONE;

    // tile_end = min(tile_start + tile_size, n)
    add.u32 %tile_end, %tile_start, %tile_size;
    min.u32 %tile_end, %tile_end, %n;

    bar.sync 0;

    // Phase 1: Compute tile_scores[j-tile_start] = Q[bid] . K[j] / sqrt(d)
    // Thread 0 computes all scores for simplicity (small tiles)
    setp.ne.u32 %p_j, %tid, 0;
    @%p_j bra SCORES_DONE;

    mov.f32 %tile_max, 0fFF7FFFFF;
    mov.u32 %j, %tile_start;
SCORE_LOOP:
    setp.ge.u32 %p_j, %j, %tile_end;
    @%p_j bra SCORE_DONE;

    mov.f32 %dot, 0f00000000;
    mov.u32 %kk, 0;
DOT_LOOP:
    setp.ge.u32 %p_k, %kk, %d;
    @%p_k bra DOT_DONE;

    // Q[bid * d + kk]
    mad.lo.u32 %tmp32, %bid, %d, %kk;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %q_ptr, %off64;
    ld.global.f32 %q_val, [%addr];

    // K[j * d + kk]
    mad.lo.u32 %tmp32, %j, %d, %kk;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %k_ptr, %off64;
    ld.global.f32 %k_val, [%addr];

    fma.rn.f32 %dot, %q_val, %k_val, %dot;
    add.u32 %kk, %kk, 1;
    bra DOT_LOOP;
DOT_DONE:

    mul.f32 %score, %dot, %scale;

    // Store in shared memory
    sub.u32 %tmp32, %j, %tile_start;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %off64, tile_scores;
    st.shared.f32 [%addr], %score;

    max.f32 %tile_max, %tile_max, %score;

    add.u32 %j, %j, 1;
    bra SCORE_LOOP;
SCORE_DONE:

    // Store tile_max in scores[255] for other threads to read
    mul.wide.u32 %off64, 255, 4;
    add.u64 %addr, %off64, tile_scores;
    st.shared.f32 [%addr], %tile_max;

SCORES_DONE:
    bar.sync 0;

    // All threads read tile_max
    mul.wide.u32 %off64, 255, 4;
    add.u64 %addr, %off64, tile_scores;
    ld.shared.f32 %tile_max, [%addr];

    // new_max = max(running_max, tile_max)
    max.f32 %new_max, %running_max, %tile_max;

    // correction = exp(running_max - new_max)
    sub.f32 %correction, %running_max, %new_max;
    mul.f32 %correction, %correction, 0f3FB8AA3B;
    ex2.approx.f32 %correction, %correction;

    // Rescale accumulator and running_sum
    mul.f32 %acc, %acc, %correction;
    mul.f32 %running_sum, %running_sum, %correction;

    // Accumulate this tile: for each j in tile
    // Each thread handles its own dimension(s) for the V weighted sum
    // For simplicity, thread 0 accumulates running_sum, all threads accumulate acc for their dim
    mov.u32 %j, %tile_start;
ACC_LOOP:
    setp.ge.u32 %p_j, %j, %tile_end;
    @%p_j bra ACC_DONE;

    // Load score for this j
    sub.u32 %tmp32, %j, %tile_start;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %off64, tile_scores;
    ld.shared.f32 %score, [%addr];

    // weight = exp(score - new_max)
    sub.f32 %weight, %score, %new_max;
    mul.f32 %weight, %weight, 0f3FB8AA3B;
    ex2.approx.f32 %weight, %weight;

    // acc += weight * V[j * d + tid] (each thread handles one dimension)
    setp.ge.u32 %p_d, %tid, %d;
    @%p_d bra SKIP_V;

    mad.lo.u32 %tmp32, %j, %d, %tid;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %v_ptr, %off64;
    ld.global.f32 %v_val, [%addr];
    fma.rn.f32 %acc, %weight, %v_val, %acc;

SKIP_V:
    // Thread 0 accumulates running_sum
    setp.ne.u32 %p_d, %tid, 0;
    @%p_d bra SKIP_SUM;
    add.f32 %running_sum, %running_sum, %weight;
SKIP_SUM:

    add.u32 %j, %j, 1;
    bra ACC_LOOP;
ACC_DONE:

    mov.f32 %running_max, %new_max;

    add.u32 %tile_start, %tile_start, %tile_size;
    bra TILE_LOOP;
TILE_DONE:

    bar.sync 0;

    // Normalize: output[bid * d + tid] = acc / running_sum
    setp.ge.u32 %p_d, %tid, %d;
    @%p_d bra EXIT;

    // Broadcast running_sum from thread 0 via shared memory
    setp.ne.u32 %p_d, %tid, 0;
    @%p_d bra LOAD_SUM;
    // Thread 0 stores running_sum
    mov.u64 %addr, tile_scores;
    st.shared.f32 [%addr], %running_sum;
LOAD_SUM:
    bar.sync 0;
    mov.u64 %addr, tile_scores;
    ld.shared.f32 %rcp_sum, [%addr];
    rcp.approx.f32 %rcp_sum, %rcp_sum;

    mul.f32 %acc, %acc, %rcp_sum;

    // Store output
    mad.lo.u32 %tmp32, %bid, %d, %tid;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %out_ptr, %off64;
    st.global.f32 [%addr], %acc;

EXIT:
    ret;
}
"#
}
