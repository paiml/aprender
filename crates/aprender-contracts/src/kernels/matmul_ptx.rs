/// PTX assembly for the tiled matrix multiplication kernel.
///
/// Uses 16x16 thread blocks with shared memory tiles. Each block computes a
/// 16x16 sub-matrix of C by iterating over tiles of the K dimension.
pub fn matmul_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry matmul_kernel(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C,
    .param .u32 M,
    .param .u32 P,
    .param .u32 N
) {
    .reg .u32 %tx, %ty, %bx, %by, %row, %col;
    .reg .u32 %m, %p, %n, %tile, %kk;
    .reg .u32 %a_col, %b_row;
    .reg .u32 %as_idx, %bs_idx, %tmp32;
    .reg .u64 %a_ptr, %b_ptr, %c_ptr, %addr, %off64;
    .reg .f32 %acc, %a_val, %b_val;
    .reg .pred %p_tile, %p_a, %p_b;
    .shared .f32 As[256];
    .shared .f32 Bs[256];

    // Thread and block indices
    mov.u32 %tx, %tid.x;
    mov.u32 %ty, %tid.y;
    mov.u32 %bx, %ctaid.x;
    mov.u32 %by, %ctaid.y;

    // Load dimensions
    ld.param.u32 %m, [M];
    ld.param.u32 %p, [P];
    ld.param.u32 %n, [N];
    ld.param.u64 %a_ptr, [A];
    ld.param.u64 %b_ptr, [B];
    ld.param.u64 %c_ptr, [C];

    // row = by * 16 + ty, col = bx * 16 + tx
    shl.b32 %row, %by, 4;
    add.u32 %row, %row, %ty;
    shl.b32 %col, %bx, 4;
    add.u32 %col, %col, %tx;

    // Initialize accumulator
    mov.f32 %acc, 0f00000000;

    // Tile loop: tile = 0, 16, 32, ...
    mov.u32 %tile, 0;
TILE_LOOP:
    setp.ge.u32 %p_tile, %tile, %p;
    @%p_tile bra TILE_DONE;

    // Load A[row][tile + tx] into shared memory As[ty][tx]
    add.u32 %a_col, %tile, %tx;
    setp.lt.u32 %p_a, %row, %m;
    setp.lt.u32 %p_b, %a_col, %p;
    and.pred %p_a, %p_a, %p_b;
    // As index = ty * 16 + tx
    shl.b32 %as_idx, %ty, 4;
    add.u32 %as_idx, %as_idx, %tx;
    @!%p_a mov.f32 %a_val, 0f00000000;
    @%p_a mad.lo.u32 %tmp32, %row, %p, %a_col;
    @%p_a mul.wide.u32 %off64, %tmp32, 4;
    @%p_a add.u64 %addr, %a_ptr, %off64;
    @%p_a ld.global.f32 %a_val, [%addr];
    mul.wide.u32 %off64, %as_idx, 4;
    add.u64 %addr, %off64, As;
    st.shared.f32 [%addr], %a_val;

    // Load B[tile + ty][col] into shared memory Bs[ty][tx]
    add.u32 %b_row, %tile, %ty;
    setp.lt.u32 %p_a, %b_row, %p;
    setp.lt.u32 %p_b, %col, %n;
    and.pred %p_a, %p_a, %p_b;
    shl.b32 %bs_idx, %ty, 4;
    add.u32 %bs_idx, %bs_idx, %tx;
    @!%p_a mov.f32 %b_val, 0f00000000;
    @%p_a mad.lo.u32 %tmp32, %b_row, %n, %col;
    @%p_a mul.wide.u32 %off64, %tmp32, 4;
    @%p_a add.u64 %addr, %b_ptr, %off64;
    @%p_a ld.global.f32 %b_val, [%addr];
    mul.wide.u32 %off64, %bs_idx, 4;
    add.u64 %addr, %off64, Bs;
    st.shared.f32 [%addr], %b_val;

    bar.sync 0;

    // Accumulate: acc += As[ty][kk] * Bs[kk][tx] for kk in 0..16
    mov.u32 %kk, 0;
INNER_LOOP:
    setp.ge.u32 %p_tile, %kk, 16;
    @%p_tile bra INNER_DONE;

    // Load As[ty][kk]
    shl.b32 %as_idx, %ty, 4;
    add.u32 %as_idx, %as_idx, %kk;
    mul.wide.u32 %off64, %as_idx, 4;
    add.u64 %addr, %off64, As;
    ld.shared.f32 %a_val, [%addr];

    // Load Bs[kk][tx]
    shl.b32 %bs_idx, %kk, 4;
    add.u32 %bs_idx, %bs_idx, %tx;
    mul.wide.u32 %off64, %bs_idx, 4;
    add.u64 %addr, %off64, Bs;
    ld.shared.f32 %b_val, [%addr];

    fma.rn.f32 %acc, %a_val, %b_val, %acc;

    add.u32 %kk, %kk, 1;
    bra INNER_LOOP;
INNER_DONE:

    bar.sync 0;

    add.u32 %tile, %tile, 16;
    bra TILE_LOOP;
TILE_DONE:

    // Store C[row][col] = acc (if in bounds)
    setp.lt.u32 %p_a, %row, %m;
    setp.lt.u32 %p_b, %col, %n;
    and.pred %p_a, %p_a, %p_b;
    @!%p_a bra EXIT;

    mad.lo.u32 %tmp32, %row, %n, %col;
    mul.wide.u32 %off64, %tmp32, 4;
    add.u64 %addr, %c_ptr, %off64;
    st.global.f32 [%addr], %acc;

EXIT:
    ret;
}
"#
}
