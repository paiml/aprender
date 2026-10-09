// #3953 / #4952: the weight -> f32 dequants of the Qwen3.5 batched prefill.
//
// `qwen35_project_rows` dequantizes every projection weight before its GEMM, in every
// `APR_QWEN35_PREFILL_GEMM` mode, so a weight type with a GEMV but no dequant still
// refuses the GPU prefill. These are the dequants for IQ2_S, IQ2_XXS, IQ3_S, IQ3_XXS and
// Q2_K, all 256-element super-blocks, sharing one launch shape and one skeleton:
//
//   grid `(n, ceil(k/256))`, 32 threads, args `(out_ptr, w_ptr, k_dim, n_dim)`.
//   Block `(row, blk)` writes the 256 outputs of super-block `blk` of row `row`.
//   No shape is baked in, so one module per type serves every shape.
//
// Every kernel here is:
//   * BIT-EXACT against its CPU oracle: every float op is `.rn` (ptxas cannot contract
//     it into an fma) and runs in the oracle's order;
//   * LOOP-FREE: every per-thread step is unrolled at generation time, so the module has
//     no backward branch and sm_121's branch patch has nothing to rewrite;
//   * column-guarded: an output at or past `k` is not stored (predicated `st`), so a `k`
//     that is not a multiple of 256 never writes into the next row;
//   * ASCII ONLY (#3477);
//   * LAYOUT-001 row-major in and out: block `w_ptr + (row * ceil(k/256) + blk) * bytes`,
//     output `out_ptr + (row * k + col) * 4`.
//
// Lookup tables are generated from the Rust constants the CPU oracles read, never
// copied by hand, under `_dq` names of their own.

const DEQUANT_PTX_HEADER: &str = r"
.version 7.5
.target sm_70
.address_size 64
";

/// A generated `.global` table: `decl` is e.g. `.align 4 .u32 iq3s_dq_grid[512]`, the
/// values are written eight per line.
fn dequant_ptx_table<T: std::fmt::Display>(decl: &str, vals: impl IntoIterator<Item = T>) -> String {
    let mut body = String::new();
    for (i, v) in vals.into_iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        body.push_str(if i % 8 == 0 { "\n    " } else { " " });
        body.push_str(&v.to_string());
    }
    format!("\n.global {decl} = {{{body}\n}};\n")
}

/// `KSIGNS_IQ2XS` (the 7-bit sign code -> 8 sign bits table of IQ2_XXS and IQ3_XXS).
fn ksigns_iq2xs_dq_table() -> String {
    dequant_ptx_table(
        ".align 1 .u8 ksigns_iq2xs_dq[128]",
        crate::quantize::iq_grids::KSIGNS_IQ2XS,
    )
}

/// Entry, parameters, the row and block guards, and the block pointer. Registers the
/// body may rely on: `%r0` tid, `%r1` row, `%r2` blk, `%r3` k, `%r4` n, `%r5` nb,
/// `%rd0` out_ptr, `%rd1` w_ptr, `%rd2` the block.
fn dequant_ptx_prologue(entry: &str, exit: &str, block_bytes: u32) -> String {
    format!(
        r"
.visible .entry {entry}(
    .param .u64 out_ptr,
    .param .u64 w_ptr,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{{
    .reg .u32 %r<40>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<8>;
    .reg .b16 %h<2>;
    .reg .pred %p<8>;

    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.x;               // row
    mov.u32 %r2, %ctaid.y;               // blk
    ld.param.u32 %r3, [k_dim];
    ld.param.u32 %r4, [n_dim];
    ld.param.u64 %rd0, [out_ptr];
    ld.param.u64 %rd1, [w_ptr];

    setp.ge.u32 %p0, %r1, %r4;
    @%p0 bra {exit};

    // nb = ceil(k_dim / 256)
    add.u32 %r5, %r3, 255;
    shr.u32 %r5, %r5, 8;
    setp.ge.u32 %p1, %r2, %r5;
    @%p1 bra {exit};

    // block = w_ptr + (row * nb + blk) * {block_bytes}
    mad.lo.u32 %r6, %r1, %r5, %r2;
    mul.wide.u32 %rd2, %r6, {block_bytes};
    add.u64 %rd2, %rd1, %rd2;
"
    )
}

/// The IQ lane and the f16 `d` at +0: `%r7` ib, `%r8` l, `%f1` d. Lane `(ib, l)` owns
/// the 8 outputs `32*ib + 8*l ..+8` of the super-block, as in the IQ GEMVs.
const IQ_DEQUANT_LANE_PTX: &str = r"
    // ib = tid >> 2   (0..8),  l = tid & 3   (0..4)
    shr.u32 %r7, %r0, 2;
    and.b32 %r8, %r0, 3;

    // d (f16 at +0)
    ld.global.b16 %h0, [%rd2];
    cvt.f32.f16 %f1, %h0;
";

/// IQ2_XXS and IQ3_XXS: the sign byte `KSIGNS_IQ2XS[(aux >> 7l) & 127]` into `%r15`,
/// from the sub-block word `aux` in `%r11`.
const IQ_KSIGNS_SIGN_BYTE_PTX: &str = r"
    // sign byte = KSIGNS_IQ2XS[(aux >> 7l) & 127]
    mul.lo.u32 %r14, %r8, 7;
    shr.u32 %r14, %r11, %r14;
    and.b32 %r14, %r14, 127;
    mov.u64 %rd5, ksigns_iq2xs_dq;
    cvt.u64.u32 %rd6, %r14;
    add.u64 %rd6, %rd5, %rd6;
    ld.global.u8 %r15, [%rd6];
";

/// IQ3_XXS and IQ3_S: the two u32 grid entries indexed by `%r16` and `%r17` into
/// `%r18` (its bytes are columns 0..4) and `%r19` (columns 4..8).
fn iq3_grid_pair_ptx(table: &str) -> String {
    format!(
        r"
    // g1 -> columns 0..4, g2 -> columns 4..8
    mov.u64 %rd5, {table};
    mul.wide.u32 %rd6, %r16, 4;
    add.u64 %rd6, %rd5, %rd6;
    ld.global.u32 %r18, [%rd6];
    mul.wide.u32 %rd6, %r17, 4;
    add.u64 %rd6, %rd5, %rd6;
    ld.global.u32 %r19, [%rd6];
"
    )
}

/// The IQ store tail. In: `%f3` db, `%r18` and `%r19` the magnitude words (byte `j` of
/// `%r18` is column `j`, byte `j` of `%r19` is column `j + 4`), `%r15` the sign byte
/// (bit `c` negates column `c`). Lane `(ib, l)` stores `db * m`, negated by its sign
/// bit, at `col0 = blk*256 + 32*ib + 8*l` onwards, never at a column `>= k`.
fn iq_dequant_store_tail(exit: &str) -> String {
    let mut ptx = String::from(
        r"
    // col0 = blk*256 + 32*ib + 8*l
    shl.b32 %r20, %r2, 8;
    shl.b32 %r21, %r7, 5;
    add.u32 %r20, %r20, %r21;
    shl.b32 %r21, %r8, 3;
    add.u32 %r20, %r20, %r21;

    // dst = out_ptr + (row * k + col0) * 4
    mul.wide.u32 %rd7, %r1, %r3;
    cvt.u64.u32 %rd8, %r20;
    add.u64 %rd7, %rd7, %rd8;
    shl.b64 %rd7, %rd7, 2;
    add.u64 %rd7, %rd0, %rd7;
",
    );
    for j in 0..4u32 {
        let (sh, hi_bit, lo_off, hi_off) = (8 * j, j + 4, 4 * j, 4 * (j + 4));
        ptx.push_str(&format!(
            r"
    // j = {j}: byte {j} of the lo word -> col0+{j} (sign bit {j}), byte {j} of the hi word -> col0+{hi_bit} (sign bit {hi_bit})
    shr.u32 %r22, %r18, {sh};
    and.b32 %r22, %r22, 255;             // m_lo
    shr.u32 %r23, %r19, {sh};
    and.b32 %r23, %r23, 255;             // m_hi
    shr.u32 %r24, %r15, {j};
    and.b32 %r24, %r24, 1;
    shr.u32 %r25, %r15, {hi_bit};
    and.b32 %r25, %r25, 1;
    cvt.rn.f32.u32 %f4, %r22;
    mul.rn.f32 %f4, %f3, %f4;            // db * m_lo
    setp.ne.u32 %p2, %r24, 0;
    @%p2 neg.f32 %f4, %f4;
    cvt.rn.f32.u32 %f5, %r23;
    mul.rn.f32 %f5, %f3, %f5;            // db * m_hi
    setp.ne.u32 %p3, %r25, 0;
    @%p3 neg.f32 %f5, %f5;
    add.u32 %r26, %r20, {j};
    setp.lt.u32 %p4, %r26, %r3;
    @%p4 st.global.f32 [%rd7+{lo_off}], %f4;
    add.u32 %r27, %r20, {hi_bit};
    setp.lt.u32 %p5, %r27, %r3;
    @%p5 st.global.f32 [%rd7+{hi_off}], %f5;
"
        ));
    }
    ptx.push_str(&format!("\n{exit}:\n    ret;\n}}\n"));
    ptx
}

/// #3953: IQ2_S (GGML type 22), 82-byte super-blocks: `d` f16 at +0, `qs[32]` at +2,
/// `signs[32]` at +34, `qh[8]` at +66, `scales[8]` at +74. Follows
/// `quantize::iq2_s::dequantize_iq2_s_block`, as the IQ2_S GEMV does line for line:
/// `idx = qs[4ib+l] | ((qh[ib] >> 2l) & 3) << 8`, `db = (d * (0.5 + nib)) * 0.25` with
/// the nibble of `scales[ib]` chosen by `l >> 1`, then `db * m`, negated by the sign bit.
/// The grid table is the GEMV's, `iq2s_grid_g`.
fn generate_iq2_s_dequant_ptx() -> String {
    const EXIT: &str = "$L_2sd_exit";
    let mut ptx = String::from(DEQUANT_PTX_HEADER);
    ptx.push_str(
        r"
// IQ2S_GRID: generated from quantize::iq_grids::IQ2S_GRID, 1024 u64 entries as
// 2048 little-endian u32 words (lo, hi).
.global .align 8 .u32 iq2s_grid_g[2048] = {",
    );
    ptx.push_str(&iq2s_grid_ptx_words());
    ptx.push_str("\n};\n");
    ptx.push_str(&dequant_ptx_prologue("iq2_s_dequant_to_f32", EXIT, 82));
    ptx.push_str(IQ_DEQUANT_LANE_PTX);
    ptx.push_str(
        r"
    // sc = scales[ib] (at +74), qh_byte = qh[ib] (at +66)
    cvt.u64.u32 %rd3, %r7;
    add.u64 %rd3, %rd2, %rd3;
    ld.global.u8 %r9, [%rd3+74];
    ld.global.u8 %r10, [%rd3+66];

    // nibble: l >> 1 selects it -- low for l = 0,1, high for l = 2,3
    shr.u32 %r11, %r8, 1;
    shl.b32 %r11, %r11, 2;               // 0 or 4
    shr.u32 %r12, %r9, %r11;
    and.b32 %r12, %r12, 15;

    // db = (d * (0.5 + nib)) * 0.25, in the reference's order
    cvt.rn.f32.u32 %f2, %r12;
    add.rn.f32 %f2, %f2, 0f3F000000;     // + 0.5
    mul.rn.f32 %f3, %f1, %f2;            // d * (0.5 + nib)
    mul.rn.f32 %f3, %f3, 0f3E800000;     // * 0.25  -> db

    // lane = 4*ib + l: idx low 8 bits = qs[lane] (at +2), sign_byte = signs[lane] (at +34)
    shl.b32 %r13, %r7, 2;
    add.u32 %r13, %r13, %r8;
    cvt.u64.u32 %rd4, %r13;
    add.u64 %rd4, %rd2, %rd4;
    ld.global.u8 %r14, [%rd4+2];
    ld.global.u8 %r15, [%rd4+34];

    // idx high 2 bits = (qh_byte >> 2l) & 3, placed at bit 8
    shl.b32 %r16, %r8, 1;                // 2l
    shr.u32 %r17, %r10, %r16;
    and.b32 %r17, %r17, 3;
    shl.b32 %r17, %r17, 8;
    or.b32 %r14, %r14, %r17;             // idx, 0..1023

    // grid entry = 8 bytes: lo word bytes 0..3, hi word bytes 4..7
    mov.u64 %rd5, iq2s_grid_g;
    mul.wide.u32 %rd6, %r14, 8;
    add.u64 %rd6, %rd5, %rd6;
    ld.global.u32 %r18, [%rd6];          // g_lo
    ld.global.u32 %r19, [%rd6+4];        // g_hi
",
    );
    ptx.push_str(&iq_dequant_store_tail(EXIT));
    ptx
}

/// #4952: IQ2_XXS (GGML type 16), 66-byte super-blocks: `d` f16 at +0, then per
/// 32-element sub-block `ib` eight bytes at `2 + 8*ib`: four grid indices (`aux0`), then
/// `aux1` = four 7-bit sign codes and a 4-bit scale. Follows
/// `quantize::iq2_xxs::dequantize_iq2_xxs_block`: `db = (d * (0.5 + (aux1 >> 28))) * 0.25`
/// evaluated left to right as the reference writes it, grid
/// `IQ2XXS_GRID[byte l of aux0]`, signs `KSIGNS_IQ2XS[(aux1 >> 7l) & 127]`.
fn generate_iq2_xxs_dequant_ptx() -> String {
    const EXIT: &str = "$L_2xd_exit";
    let grid = &crate::quantize::iq_grids::IQ2XXS_GRID;
    let mut ptx = String::from(DEQUANT_PTX_HEADER);
    ptx.push_str("\n// IQ2XXS_GRID: 256 u64 entries as 512 little-endian u32 words (lo, hi).");
    ptx.push_str(&dequant_ptx_table(
        ".align 8 .u32 iq2xxs_dq_grid[512]",
        grid.iter().flat_map(|&v| [v & 0xffff_ffff, v >> 32]),
    ));
    ptx.push_str(&ksigns_iq2xs_dq_table());
    ptx.push_str(&dequant_ptx_prologue("iq2_xxs_dequant_to_f32", EXIT, 66));
    ptx.push_str(IQ_DEQUANT_LANE_PTX);
    ptx.push_str(
        r"
    // the sub-block's 8 bytes start at block + 2 + 8*ib: grid index = byte l of aux0;
    // aux1 (u32 at +6 + 8*ib) is only 2-byte aligned, so it is read as two u16
    shl.b32 %r9, %r7, 3;
    cvt.u64.u32 %rd3, %r9;
    add.u64 %rd3, %rd2, %rd3;
    cvt.u64.u32 %rd4, %r8;
    add.u64 %rd4, %rd3, %rd4;
    ld.global.u8 %r10, [%rd4+2];         // grid index
    ld.global.u16 %r11, [%rd3+6];
    ld.global.u16 %r12, [%rd3+8];
    shl.b32 %r12, %r12, 16;
    or.b32 %r11, %r11, %r12;             // aux1

    // db = (d * (0.5 + (aux1 >> 28))) * 0.25, in the reference's order
    shr.u32 %r13, %r11, 28;
    cvt.rn.f32.u32 %f2, %r13;
    add.rn.f32 %f2, %f2, 0f3F000000;     // + 0.5
    mul.rn.f32 %f3, %f1, %f2;            // d * (0.5 + s)
    mul.rn.f32 %f3, %f3, 0f3E800000;     // * 0.25  -> db
",
    );
    ptx.push_str(IQ_KSIGNS_SIGN_BYTE_PTX);
    ptx.push_str(
        r"
    // grid entry = 8 bytes: lo word -> columns 0..4, hi word -> columns 4..8
    mov.u64 %rd5, iq2xxs_dq_grid;
    mul.wide.u32 %rd6, %r10, 8;
    add.u64 %rd6, %rd5, %rd6;
    ld.global.u32 %r18, [%rd6];
    ld.global.u32 %r19, [%rd6+4];
",
    );
    ptx.push_str(&iq_dequant_store_tail(EXIT));
    ptx
}

/// #4952: IQ3_XXS (GGML type 18), 98-byte super-blocks: `d` f16 at +0, `qs[64]` grid
/// indices at +2, then one u32 per sub-block at `66 + 4*ib` holding four 7-bit sign
/// codes and a 4-bit scale. Follows `quantize::iq3_xxs::dequantize_iq3_xxs_block`:
/// `db = (d * (0.5 + (aux >> 28))) * 0.5` left to right, grids
/// `IQ3XXS_GRID[qs[8ib + 2l]]` (columns 0..4) and `IQ3XXS_GRID[qs[8ib + 2l + 1]]`
/// (columns 4..8), signs `KSIGNS_IQ2XS[(aux >> 7l) & 127]`.
fn generate_iq3_xxs_dequant_ptx() -> String {
    const EXIT: &str = "$L_3xd_exit";
    let mut ptx = String::from(DEQUANT_PTX_HEADER);
    ptx.push_str("\n// IQ3XXS_GRID: 256 u32 entries.");
    ptx.push_str(&dequant_ptx_table(
        ".align 4 .u32 iq3xxs_dq_grid[256]",
        crate::quantize::iq_grids::IQ3XXS_GRID,
    ));
    ptx.push_str(&ksigns_iq2xs_dq_table());
    ptx.push_str(&dequant_ptx_prologue("iq3_xxs_dequant_to_f32", EXIT, 98));
    ptx.push_str(IQ_DEQUANT_LANE_PTX);
    ptx.push_str(
        r"
    // aux = u32 at +66 + 4*ib, only 2-byte aligned, so read as two u16
    shl.b32 %r9, %r7, 2;
    cvt.u64.u32 %rd3, %r9;
    add.u64 %rd3, %rd2, %rd3;
    ld.global.u16 %r11, [%rd3+66];
    ld.global.u16 %r12, [%rd3+68];
    shl.b32 %r12, %r12, 16;
    or.b32 %r11, %r11, %r12;             // aux

    // db = (d * (0.5 + (aux >> 28))) * 0.5, in the reference's order
    shr.u32 %r13, %r11, 28;
    cvt.rn.f32.u32 %f2, %r13;
    add.rn.f32 %f2, %f2, 0f3F000000;     // + 0.5
    mul.rn.f32 %f3, %f1, %f2;            // d * (0.5 + s)
    mul.rn.f32 %f3, %f3, 0f3F000000;     // * 0.5  -> db
",
    );
    ptx.push_str(IQ_KSIGNS_SIGN_BYTE_PTX);
    ptx.push_str(
        r"
    // grid indices qs[8*ib + 2l] and qs[8*ib + 2l + 1] (qs at +2)
    shl.b32 %r9, %r7, 3;
    shl.b32 %r10, %r8, 1;
    add.u32 %r9, %r9, %r10;
    cvt.u64.u32 %rd4, %r9;
    add.u64 %rd4, %rd2, %rd4;
    ld.global.u8 %r16, [%rd4+2];
    ld.global.u8 %r17, [%rd4+3];
",
    );
    ptx.push_str(&iq3_grid_pair_ptx("iq3xxs_dq_grid"));
    ptx.push_str(&iq_dequant_store_tail(EXIT));
    ptx
}

/// #4952: IQ3_S (GGML type 21), 110-byte super-blocks: `d` f16 at +0, `qs[64]` at +2,
/// `qh[8]` at +66, `signs[32]` at +74, `scales[4]` at +106. Follows
/// `quantize::iq3_s::dequantize_iq3_s_block`: `db = d * (1 + 2 * nib)` with the low
/// nibble of `scales[ib >> 1]` for even `ib` and the high one for odd, 9-bit grid
/// indices `qs[8ib + 2l] | ((qh[ib] << (8 - 2l)) & 256)` and
/// `qs[8ib + 2l + 1] | ((qh[ib] << (7 - 2l)) & 256)` -- bits `2l` and `2l + 1` of
/// `qh[ib]` placed at bit 8 -- and the explicit sign byte `signs[4ib + l]`.
fn generate_iq3_s_dequant_ptx() -> String {
    const EXIT: &str = "$L_3sd_exit";
    let mut ptx = String::from(DEQUANT_PTX_HEADER);
    ptx.push_str("\n// IQ3S_GRID: 512 u32 entries.");
    ptx.push_str(&dequant_ptx_table(
        ".align 4 .u32 iq3s_dq_grid[512]",
        crate::quantize::iq_grids::IQ3S_GRID,
    ));
    ptx.push_str(&dequant_ptx_prologue("iq3_s_dequant_to_f32", EXIT, 110));
    ptx.push_str(IQ_DEQUANT_LANE_PTX);
    ptx.push_str(
        r"
    // nib = scales[ib >> 1] (at +106): low nibble for even ib, high for odd
    shr.u32 %r9, %r7, 1;
    cvt.u64.u32 %rd3, %r9;
    add.u64 %rd3, %rd2, %rd3;
    ld.global.u8 %r10, [%rd3+106];
    and.b32 %r11, %r7, 1;
    shl.b32 %r11, %r11, 2;               // 0 or 4
    shr.u32 %r12, %r10, %r11;
    and.b32 %r12, %r12, 15;

    // db = d * (1 + 2 * nib), in the reference's order
    cvt.rn.f32.u32 %f2, %r12;
    mul.rn.f32 %f2, %f2, 0f40000000;     // 2 * nib
    add.rn.f32 %f2, %f2, 0f3F800000;     // 1 + 2 * nib
    mul.rn.f32 %f3, %f1, %f2;            // db

    // qh_byte = qh[ib] (at +66); sign byte = signs[4*ib + l] (at +74)
    cvt.u64.u32 %rd3, %r7;
    add.u64 %rd3, %rd2, %rd3;
    ld.global.u8 %r13, [%rd3+66];
    shl.b32 %r14, %r7, 2;
    add.u32 %r14, %r14, %r8;
    cvt.u64.u32 %rd4, %r14;
    add.u64 %rd4, %rd2, %rd4;
    ld.global.u8 %r15, [%rd4+74];

    // low 8 index bits: qs[8*ib + 2l] and qs[8*ib + 2l + 1] (qs at +2)
    shl.b32 %r9, %r7, 3;
    shl.b32 %r10, %r8, 1;                // 2l
    add.u32 %r9, %r9, %r10;
    cvt.u64.u32 %rd4, %r9;
    add.u64 %rd4, %rd2, %rd4;
    ld.global.u8 %r16, [%rd4+2];
    ld.global.u8 %r17, [%rd4+3];

    // 9th index bit: bit 2l of qh_byte for i1, bit 2l+1 for i2, placed at bit 8
    shr.u32 %r21, %r13, %r10;            // qh >> 2l
    and.b32 %r22, %r21, 1;
    shl.b32 %r22, %r22, 8;
    or.b32 %r16, %r16, %r22;             // i1
    shr.u32 %r21, %r21, 1;               // qh >> (2l + 1)
    and.b32 %r21, %r21, 1;
    shl.b32 %r21, %r21, 8;
    or.b32 %r17, %r17, %r21;             // i2
",
    );
    ptx.push_str(&iq3_grid_pair_ptx("iq3s_dq_grid"));
    ptx.push_str(&iq_dequant_store_tail(EXIT));
    ptx
}

/// #4952: Q2_K (GGML type 10), 84-byte super-blocks: `scales[16]` at +0, `qs[64]` at
/// +16, `d` f16 at +80, `dmin` f16 at +82. Follows `quantize::dequantize_q2_k` (ggml's
/// `dequantize_row_q2_K` order): for group `g` in 0..2, shift step `s` in 0..4 and half
/// `h` in 0..2, the 16 outputs at `128g + 32s + 16h ..+16` take the 2-bit field
/// `(qs[32g + 16h + i] >> 2s) & 3` and the scale byte `sc = scales[8g + 2s + h]`:
/// `(d * (sc & 15)) * q - dmin * (sc >> 4)`, a `.rn` multiply then a `.rn` subtract,
/// exactly the oracle's two roundings (no fma).
///
/// Thread `t` is `(h = t >> 4, i = t & 15)`: it reads its two `qs` bytes once and
/// writes the 8 outputs `128g + 32s + 16h + i`, unrolled over `(g, s)`.
fn generate_q2_k_dequant_ptx() -> String {
    const EXIT: &str = "$L_q2d_exit";
    let mut ptx = String::from(DEQUANT_PTX_HEADER);
    ptx.push_str(&dequant_ptx_prologue("q2_k_dequant_to_f32", EXIT, 84));
    ptx.push_str(
        r"
    // d (f16 at +80) and dmin (f16 at +82)
    ld.global.b16 %h0, [%rd2+80];
    cvt.f32.f16 %f1, %h0;
    ld.global.b16 %h1, [%rd2+82];
    cvt.f32.f16 %f2, %h1;

    // h = tid >> 4 picks the 16-byte half of each 32-byte qs window, i = tid & 15
    shr.u32 %r7, %r0, 4;
    and.b32 %r8, %r0, 15;
    shl.b32 %r9, %r7, 4;
    add.u32 %r9, %r9, %r8;               // 16h + i

    // the thread's two quant bytes qs[32g + 16h + i] (qs at +16), g = 0 and 1
    cvt.u64.u32 %rd3, %r9;
    add.u64 %rd3, %rd2, %rd3;
    ld.global.u8 %r10, [%rd3+16];        // g = 0
    ld.global.u8 %r11, [%rd3+48];        // g = 1

    // scale bytes scales[8g + 2s + h] (at +0)
    cvt.u64.u32 %rd4, %r7;
    add.u64 %rd4, %rd2, %rd4;

    // col0 = blk*256 + 16h + i;  dst = out_ptr + (row * k + col0) * 4
    shl.b32 %r20, %r2, 8;
    add.u32 %r20, %r20, %r9;
    mul.wide.u32 %rd7, %r1, %r3;
    cvt.u64.u32 %rd8, %r20;
    add.u64 %rd7, %rd7, %rd8;
    shl.b64 %rd7, %rd7, 2;
    add.u64 %rd7, %rd0, %rd7;
",
    );
    for g in 0..2u32 {
        let q = if g == 0 { "%r10" } else { "%r11" };
        for s in 0..4u32 {
            let (sc, shift, off) = (8 * g + 2 * s, 2 * s, 128 * g + 32 * s);
            let st_off = 4 * off;
            ptx.push_str(&format!(
                r"
    // g = {g}, s = {s}: scale byte {sc} (+h), bits {shift}.. of q -> col0+{off}
    ld.global.u8 %r12, [%rd4+{sc}];
    and.b32 %r13, %r12, 15;
    shr.u32 %r14, %r12, 4;
    cvt.rn.f32.u32 %f3, %r13;
    mul.rn.f32 %f3, %f1, %f3;            // dl = d * (sc & 15)
    cvt.rn.f32.u32 %f4, %r14;
    mul.rn.f32 %f4, %f2, %f4;            // ml = dmin * (sc >> 4)
    shr.u32 %r15, {q}, {shift};
    and.b32 %r15, %r15, 3;
    cvt.rn.f32.u32 %f5, %r15;
    mul.rn.f32 %f5, %f3, %f5;            // dl * q
    sub.rn.f32 %f5, %f5, %f4;            // - ml
    add.u32 %r16, %r20, {off};
    setp.lt.u32 %p2, %r16, %r3;
    @%p2 st.global.f32 [%rd7+{st_off}], %f5;
"
            ));
        }
    }
    ptx.push_str(&format!("\n{EXIT}:\n    ret;\n}}\n"));
    ptx
}

impl CudaKernels {
    /// #3953 / #4952: the prefill dequants' entry names. Their own arm, so no existing
    /// name function grows.
    fn prefill_dequant_kernel_name(kernel_type: &KernelType) -> Option<&'static str> {
        Some(match kernel_type {
            KernelType::Iq2SDequant { .. } => "iq2_s_dequant_to_f32",
            KernelType::Iq2XxsDequant { .. } => "iq2_xxs_dequant_to_f32",
            KernelType::Iq3SDequant { .. } => "iq3_s_dequant_to_f32",
            KernelType::Iq3XxsDequant { .. } => "iq3_xxs_dequant_to_f32",
            KernelType::Q2KDequant { .. } => "q2_k_dequant_to_f32",
            _ => return None,
        })
    }

    /// #3953 / #4952: the prefill dequants' PTX. Shape-free: `k` and `n` are kernel
    /// arguments, so the variant's fields never reach the module.
    fn generate_prefill_dequant_ptx(kernel_type: &KernelType) -> Option<String> {
        Some(match kernel_type {
            KernelType::Iq2SDequant { .. } => generate_iq2_s_dequant_ptx(),
            KernelType::Iq2XxsDequant { .. } => generate_iq2_xxs_dequant_ptx(),
            KernelType::Iq3SDequant { .. } => generate_iq3_s_dequant_ptx(),
            KernelType::Iq3XxsDequant { .. } => generate_iq3_xxs_dequant_ptx(),
            KernelType::Q2KDequant { .. } => generate_q2_k_dequant_ptx(),
            _ => return None,
        })
    }
}
