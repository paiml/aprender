//! Tensor-core Q4K×Q8_1 GEMM (MMQ) for prefill (#4376)
//!
//! C[M,N] = A_q8[M,K] × W_q4k[N,K]^T on `mma.sync.m16n8k32.s32.s8.s8.s32`.
//!
//! Same inputs and the same integer arithmetic as [`super::Dp4aQ4KGemmKernel`]:
//! Q8_1 activations (32 s8 + f16 d + f16 s, 36 B) and raw Q4K nibbles (0..15, which
//! are valid s8). Per Q4K sub-block j (32 values) the integer dot `Σ q·a` comes
//! from one mma, and the sub-block's scales are applied in f32:
//!
//! ```text
//! y[m,n] += d8[m,j] · ( d[n]·sc[n,j] · Σ q·a  −  dmin[n]·mn[n,j] · Σ a )
//! ```
//!
//! `Σ a` is summed exactly with dp4a, as the DP4A kernel does, so the two kernels
//! agree up to f32 summation order.
//!
//! ## Layout (v2: block tiles staged in shared memory)
//!
//! ```text
//! Grid:  (ceil(N/128), ceil(M/64))     Block: 8 warps = 2 (M) × 4 (N)
//! Warp:  32 rows × 32 cols = 2 m-tiles of 16 × 4 n-tiles of 8
//! Lane:  gid = lane/4, tig = lane%4 (PTX m16n8k32 fragment roles)
//!   A a0/a2: row gid,   k tig*4.. / 16+tig*4..     a1/a3: row gid+8
//!   B b0/b1: col gid,   k tig*4.. / 16+tig*4..
//!   C c0/c1: row gid,   col tig*2, tig*2+1         c2/c3: row gid+8
//! ```
//!
//! Per super-block the block stages, then syncs:
//!
//! - A: 64 rows × 8 Q8_1 blocks. The qs go to smem at a 272-byte row stride (68
//!   words, so the 8 `gid` rows of a fragment load hit 8 different banks), plus per
//!   (sub-block, row) `d8` and `d8·Σa` in f32, with `Σa` summed exactly by dp4a.
//! - W: the 128 raw 144-byte Q4K super-blocks at a 144-byte stride (36 words: again
//!   8 banks apart per `gid`), plus per (sub-block, col) `d·sc` and `−dmin·m`,
//!   unpacked once here instead of once per warp per sub-block.
//!
//! The mma accumulator is seeded with `0x4B40_0000` (1.5·2^23), so its s32 result
//! read as f32 is `1.5·2^23 + Σq·a` exactly (|Σq·a| ≤ 32·15·127 < 2^22), and one f32
//! subtract replaces a quarter-rate `cvt.f32.s32`.
//!
//! Q4K `qs` holds sub-blocks in pairs: bytes `32p..32p+32` carry sub-block `2p` in
//! the low nibbles and `2p+1` in the high nibbles, so a B fragment is one aligned
//! u32 load plus a mask. Out-of-range rows and columns are clamped for loads and
//! skipped for stores.

use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory, PtxSync};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// mma n-tiles per warp.
const N_TILES: u32 = 4;
/// mma m-tiles per warp.
const M_TILES: u32 = 2;
/// Warps along M and along N.
const WARPS_M: u32 = 2;
const WARPS_N: u32 = 4;
/// Columns per block.
pub const MMA_Q4K_TILE_N: u32 = 8 * N_TILES * WARPS_N;
/// Rows per block.
pub const MMA_Q4K_TILE_M: u32 = 16 * M_TILES * WARPS_M;
/// Threads per block.
pub const MMA_Q4K_THREADS: u32 = 32 * WARPS_M * WARPS_N;
/// Q8_1 block bytes (32 s8 + f16 d + f16 s).
const Q8_BLOCK_BYTES: u32 = 36;

// Shared-memory map, bytes.
const A_ROW: u32 = 272;
const A_QS: u32 = 0;
/// `[j][row/16][row%8][(row/8)%2]{d8, d8·Σa}` f32: one v4 load gives a thread both
/// of its rows (gid, gid+8) of an m-tile.
const A_SC: u32 = A_QS + MMA_Q4K_TILE_M * A_ROW;
/// `[j][col/2]{d·sc even, d·sc odd, −dmin·m even, −dmin·m odd}` f32: one v4 load
/// gives a thread both of its columns of an n-tile.
const W_RAW: u32 = A_SC + 8 * MMA_Q4K_TILE_M * 8; // [col] 144 B
const W_SC: u32 = W_RAW + MMA_Q4K_TILE_N * Q4K_SUPER_BLOCK_BYTES;
const SMEM_BYTES: u32 = W_SC + 8 * MMA_Q4K_TILE_N * 8;
/// 16-byte chunks per Q4K super-block (144 / 16).
const W_CHUNKS: u32 = Q4K_SUPER_BLOCK_BYTES / 16;

/// Tensor-core Q4K×Q8_1 GEMM (#4376). Requires sm_80+ and Q8_1 activations from
/// `Q8QuantizeKernel`; K must be a multiple of 256.
///
/// # Provable Contracts
///
/// - **C1**: Per sub-block, the integer dot equals the DP4A kernel's `Σ q·a`
/// - **C2**: The min term uses the exact integer `Σ a` (dp4a)
/// - **C3**: Output C[m, n] f32 row-major; rows ≥ M and cols ≥ N are never stored
pub struct MmaQ4KGemmKernel {
    /// M dimension (rows / tokens)
    pub m: u32,
    /// N dimension (output features)
    pub n: u32,
    /// K dimension (multiple of 256)
    pub k: u32,
}

impl MmaQ4KGemmKernel {
    /// Create a new tensor-core Q4K GEMM kernel.
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    /// Launch grid `(x, y)` for this shape.
    #[must_use]
    pub const fn grid(m: u32, n: u32) -> (u32, u32) {
        (n.div_ceil(MMA_Q4K_TILE_N), m.div_ceil(MMA_Q4K_TILE_M))
    }
}

impl Kernel for MmaQ4KGemmKernel {
    fn name(&self) -> &str {
        "mma_q4k_gemm"
    }

    #[allow(clippy::too_many_lines)]
    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("mma_q4k_gemm")
            .param(PtxType::U64, "y_ptr") // C[M, N] f32
            .param(PtxType::U64, "w_ptr") // Q4K W[N, K/256, 144B]
            .param(PtxType::U64, "q8_ptr") // Q8_1 A[M, K/32, 36B]
            .param(PtxType::U32, "m_dim")
            .param(PtxType::U32, "n_dim")
            .param(PtxType::U32, "k_dim")
            .shared_memory(SMEM_BYTES as usize)
            .build(move |ctx| {
                let block_x = ctx.special_reg(PtxReg::CtaIdX);
                let block_y = ctx.special_reg(PtxReg::CtaIdY);
                let tid = ctx.special_reg(PtxReg::TidX);
                let lane = ctx.and_u32_imm(tid, 31);
                let warp = ctx.shr_u32_imm(tid, 5);
                let gid = ctx.shr_u32_imm(lane, 2);
                let tig = ctx.and_u32_imm(lane, 3);
                let wm = ctx.and_u32_imm(warp, WARPS_M - 1);
                let wn = ctx.shr_u32_imm(warp, 1);

                let m_dim = ctx.load_param_u32("m_dim");
                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let q8_ptr = ctx.load_param_u64("q8_ptr");

                let bm = ctx.mul_u32(block_y, MMA_Q4K_TILE_M);
                let bn = ctx.mul_u32(block_x, MMA_Q4K_TILE_N);
                let num_sb = ctx.div_u32(k_dim, Q4K_SUPER_BLOCK_SIZE);
                let w_row_bytes = ctx.mul_u32(num_sb, Q4K_SUPER_BLOCK_BYTES);
                let q8_row_bytes = ctx.mul_u32(num_sb, 8 * Q8_BLOCK_BYTES);
                let one = ctx.mov_u32_imm(1);
                let m_last = ctx.sub_u32(m_dim, one);
                let n_last = ctx.sub_u32(n_dim, one);

                // ---- Staging assignments (super-block invariant) ----
                // A: Q8_1 blocks b = tid + 256·i, row b/8, sub-block b%8.
                let mut a_ld = Vec::new();
                for i in 0..(MMA_Q4K_TILE_M * 8 / MMA_Q4K_THREADS) {
                    let b = ctx.add_u32(tid, i * MMA_Q4K_THREADS);
                    let row = ctx.shr_u32_imm(b, 3);
                    let jb = ctx.and_u32_imm(b, 7);
                    let r = ctx.add_u32_reg(bm, row);
                    let r = ctx.min_u32(r, m_last);
                    let g = ctx.mul_wide_u32_reg(r, q8_row_bytes);
                    let g = ctx.add_u64(q8_ptr, g);
                    let jo = ctx.mul_wide_u32(jb, Q8_BLOCK_BYTES);
                    let g = ctx.add_u64(g, jo);
                    let s_qs = ctx.mul_u32(row, A_ROW);
                    let jq = ctx.shl_u32_imm(jb, 5);
                    let s_qs = ctx.add_u32_reg(s_qs, jq);
                    // A_SC slot of `row`: (row/16)·128 + (row%8)·16 + ((row/8)%2)·8.
                    let r16 = ctx.shr_u32_imm(row, 4);
                    let r16 = ctx.shl_u32_imm(r16, 7);
                    let r8 = ctx.and_u32_imm(row, 7);
                    let r8 = ctx.shl_u32_imm(r8, 4);
                    let h8 = ctx.shr_u32_imm(row, 3);
                    let h8 = ctx.and_u32_imm(h8, 1);
                    let h8 = ctx.shl_u32_imm(h8, 3);
                    let s_d = ctx.mul_u32(jb, MMA_Q4K_TILE_M * 8);
                    let s_d = ctx.add_u32_reg(s_d, r16);
                    let s_d = ctx.add_u32_reg(s_d, r8);
                    let s_d = ctx.add_u32_reg(s_d, h8);
                    a_ld.push((g, s_qs, s_d));
                }
                // W raw: 16-byte chunks q = tid + 256·i of the tile's 128 × 9, col q/9,
                // chunk q%9, copied global→shared by cp.async. The last round is partial.
                let mut w_ld = Vec::new();
                for i in 0..(MMA_Q4K_TILE_N * W_CHUNKS).div_ceil(MMA_Q4K_THREADS) {
                    let q = ctx.add_u32(tid, i * MMA_Q4K_THREADS);
                    let col = ctx.div_u32(q, W_CHUNKS);
                    let cq = ctx.mul_u32(col, W_CHUNKS);
                    let ch = ctx.sub_u32(q, cq);
                    let c = ctx.add_u32_reg(bn, col);
                    let c = ctx.min_u32(c, n_last);
                    let o = ctx.mul_wide_u32_reg(c, w_row_bytes);
                    let g = ctx.add_u64(w_ptr, o);
                    let co = ctx.mul_wide_u32(ch, 16);
                    let g = ctx.add_u64(g, co);
                    let s = ctx.shl_u32_imm(q, 4);
                    let s = ctx.add_u32(s, W_RAW);
                    let partial = (i + 1) * MMA_Q4K_THREADS > MMA_Q4K_TILE_N * W_CHUNKS;
                    w_ld.push((g, s, partial));
                }
                let w_last = ctx.setp_lt_u32_imm(
                    tid,
                    MMA_Q4K_TILE_N * W_CHUNKS % MMA_Q4K_THREADS,
                );
                // Scale unpack: threads < 128 own column tid, reading its header from smem.
                let own_scales = ctx.setp_lt_u32_imm(tid, MMA_Q4K_TILE_N);
                let hdr_s = {
                    let s = ctx.mul_u32(tid, Q4K_SUPER_BLOCK_BYTES);
                    ctx.add_u32(s, W_RAW)
                };
                // W_SC slot of column tid: (tid/2)·16 + (tid%2)·4.
                let sc_s = {
                    let p = ctx.shr_u32_imm(tid, 1);
                    let p = ctx.shl_u32_imm(p, 4);
                    let o = ctx.and_u32_imm(tid, 1);
                    let o = ctx.shl_u32_imm(o, 2);
                    ctx.add_u32_reg(p, o)
                };

                // ---- Fragment bases in smem (ldmatrix: lane l addresses row l%8 of
                // 8×8 matrix l/8) ----
                let l8 = ctx.and_u32_imm(lane, 7);
                let lm1 = ctx.shr_u32_imm(lane, 3);
                let lm1 = ctx.and_u32_imm(lm1, 1);
                let lm2 = ctx.shr_u32_imm(lane, 4);
                // A: matrices (rows 0-7, k 0-15), (8-15, 0-15), (0-7, 16-31), (8-15, 16-31)
                // are a0..a3 of m16n8k32.
                let mut a_fr = Vec::new();
                let mut d_fr = Vec::new(); // [j=0] v4 slot of rows (gid, gid+8)
                for mt in 0..M_TILES {
                    let r = ctx.mul_u32(wm, 16 * M_TILES);
                    let r = ctx.add_u32(r, mt * 16);
                    let o = ctx.shl_u32_imm(lm1, 3);
                    let rr = ctx.add_u32_reg(r, o);
                    let rr = ctx.add_u32_reg(rr, l8);
                    let q = ctx.mul_u32(rr, A_ROW);
                    let k = ctx.shl_u32_imm(lm2, 4);
                    a_fr.push(ctx.add_u32_reg(q, k));
                    let b = ctx.shl_u32_imm(r, 3); // (r/16)·128
                    let g = ctx.shl_u32_imm(gid, 4);
                    let d = ctx.add_u32_reg(b, g);
                    d_fr.push(ctx.add_u32(d, A_SC));
                }
                // B: per tile pair, matrices (tile 2p, k 0-15), (2p, 16-31), (2p+1, 0-15),
                // (2p+1, 16-31) are b0, b1 of each tile.
                let mut b_fr = Vec::new();
                let cw = ctx.mul_u32(wn, 8 * N_TILES);
                for p in 0..N_TILES / 2 {
                    let t = ctx.shl_u32_imm(lm2, 3);
                    let c = ctx.add_u32(cw, p * 16);
                    let c = ctx.add_u32_reg(c, t);
                    let c = ctx.add_u32_reg(c, l8);
                    let o = ctx.mul_u32(c, Q4K_SUPER_BLOCK_BYTES);
                    let k = ctx.shl_u32_imm(lm1, 4);
                    let o = ctx.add_u32_reg(o, k);
                    b_fr.push(ctx.add_u32(o, W_RAW + 16));
                }
                // [j=0] v4 slot of this thread's column pair in n-tile 0.
                let s_col = {
                    let c = ctx.shr_u32_imm(cw, 1);
                    let c = ctx.add_u32_reg(c, tig);
                    let c = ctx.shl_u32_imm(c, 4);
                    ctx.add_u32(c, W_SC)
                };

                let mut acc = Vec::new();
                for _ in 0..M_TILES * N_TILES * 4 {
                    acc.push(ctx.mov_f32_imm(0.0));
                }
                let ones = ctx.mov_u32_imm(0x0101_0101);
                let nib = ctx.mov_u32_imm(0x0F0F_0F0F);
                let magic_f = ctx.mov_f32_imm(12_582_912.0);
                // Loop-invariant accumulator seed: a read-only C for every mma.
                let seed = [0; 4].map(|_| ctx.mov_s32_imm(0x4B40_0000));

                let sb = ctx.mov_u32_imm(0);
                ctx.label("mma_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb, num_sb);
                ctx.branch_if(sb_done, "mma_sb_end");

                // ---- Stage W raw (async; lands while A is staged) ----
                let w_sb = ctx.mul_wide_u32(sb, Q4K_SUPER_BLOCK_BYTES);
                for (g, s, partial) in &w_ld {
                    if *partial {
                        ctx.branch_if_not(w_last, "mma_w_ld_done");
                    }
                    let g = ctx.add_u64(*g, w_sb);
                    ctx.cp_async_global_to_shared(*s, g, 16);
                }
                ctx.label("mma_w_ld_done");
                ctx.cp_async_commit_group();

                // ---- Stage A ----
                let q8_sb = ctx.mul_wide_u32(sb, 8 * Q8_BLOCK_BYTES);
                for (g, s_qs, s_d) in &a_ld {
                    let g = ctx.add_u64(*g, q8_sb);
                    let sa = ctx.mov_u32_imm(0);
                    for w in 0..8u32 {
                        let o = ctx.mov_u64_imm(u64::from(4 * w));
                        let a = ctx.add_u64(g, o);
                        let v = ctx.ld_global_u32(a);
                        ctx.dp4a_u32_s32_inplace(sa, ones, v);
                        let s = ctx.add_u32(*s_qs, 4 * w);
                        ctx.st_shared_u32(s, v);
                    }
                    let o = ctx.mov_u64_imm(32);
                    let a = ctx.add_u64(g, o);
                    let d8 = ctx.ld_global_f16(a);
                    let d8 = ctx.cvt_f32_f16(d8);
                    let saf = ctx.cvt_f32_s32(sa);
                    let dsa = ctx.mul_f32(d8, saf);
                    let s = ctx.add_u32(*s_d, A_SC);
                    ctx.st_shared_f32(s, d8);
                    let s = ctx.add_u32(*s_d, A_SC + 4);
                    ctx.st_shared_f32(s, dsa);
                }
                ctx.cp_async_wait_group(0);
                ctx.bar_sync(0);

                // ---- Unpack W scales: d·sc and −dmin·m per (sub-block, col) ----
                ctx.branch_if_not(own_scales, "mma_sc_done");
                {
                    let d = ctx.ld_shared_f16(hdr_s);
                    let d = ctx.cvt_f32_f16(d);
                    let a = ctx.add_u32(hdr_s, 2);
                    let dmin = ctx.ld_shared_f16(a);
                    let dmin = ctx.cvt_f32_f16(dmin);
                    let ndmin = ctx.neg_f32(dmin);
                    let mut w = [one; 3];
                    for (i, wi) in w.iter_mut().enumerate() {
                        let a = ctx.add_u32(hdr_s, 4 + 4 * i as u32);
                        *wi = ctx.ld_shared_u32(a);
                    }
                    for j in 0..8u32 {
                        let (sc, mn) = if j < 4 {
                            (ctx.bfe_u32(w[0], 8 * j, 6), ctx.bfe_u32(w[1], 8 * j, 6))
                        } else {
                            let jj = j - 4;
                            let sl = ctx.bfe_u32(w[2], 8 * jj, 4);
                            let sh = ctx.bfe_u32(w[0], 8 * jj + 6, 2);
                            let sh = ctx.shl_u32_imm(sh, 4);
                            let ml = ctx.bfe_u32(w[2], 8 * jj + 4, 4);
                            let mh = ctx.bfe_u32(w[1], 8 * jj + 6, 2);
                            let mh = ctx.shl_u32_imm(mh, 4);
                            (ctx.or_u32(sl, sh), ctx.or_u32(ml, mh))
                        };
                        let sc = ctx.cvt_f32_u32(sc);
                        let mn = ctx.cvt_f32_u32(mn);
                        let dsc = ctx.mul_f32(d, sc);
                        let ndm = ctx.mul_f32(ndmin, mn);
                        let s = ctx.add_u32(sc_s, W_SC + j * MMA_Q4K_TILE_N * 8);
                        ctx.st_shared_f32(s, dsc);
                        let s = ctx.add_u32(sc_s, W_SC + j * MMA_Q4K_TILE_N * 8 + 8);
                        ctx.st_shared_f32(s, ndm);
                    }
                }
                ctx.label("mma_sc_done");
                ctx.bar_sync(0);

                // ---- Compute ----
                for j in 0..8u32 {
                    let mut a_frag = Vec::new();
                    let mut d8 = Vec::new();
                    let mut dsa = Vec::new();
                    for mt in 0..M_TILES as usize {
                        let p = ctx.add_u32(a_fr[mt], j * 32);
                        a_frag.push(ctx.ldmatrix_x4(p));
                        let p = ctx.add_u32(d_fr[mt], j * MMA_Q4K_TILE_M * 8);
                        let v = ctx.ld_shared_f32_v4(p);
                        // v = {d8(gid), dsa(gid), d8(gid+8), dsa(gid+8)}
                        d8.extend([v[0], v[2]]);
                        dsa.extend([v[1], v[3]]);
                    }
                    let mut b_frag = Vec::new();
                    for bp in &b_fr {
                        let p = ctx.add_u32(*bp, 32 * (j / 2));
                        let m = ctx.ldmatrix_x4(p);
                        b_frag.push([m[0], m[1]]);
                        b_frag.push([m[2], m[3]]);
                    }
                    for (t, b) in b_frag.iter_mut().enumerate() {
                        for v in b.iter_mut() {
                            if j % 2 == 1 {
                                *v = ctx.shr_u32_imm(*v, 4);
                            }
                            *v = ctx.and_u32(*v, nib);
                        }
                        let b = *b;
                        let p = ctx.add_u32(s_col, j * MMA_Q4K_TILE_N * 8 + t as u32 * 64);
                        let v = ctx.ld_shared_f32_v4(p);
                        let (dsc, ndm) = ([v[0], v[1]], [v[2], v[3]]);
                        for mt in 0..M_TILES as usize {
                            let c = ctx.mma_sync_m16n8k32_s8(&a_frag[mt], &b, &seed);
                            for h in 0..2usize {
                                for col in 0..2usize {
                                    // c[2h + col] is (row gid + 8h, col tig*2 + col).
                                    let f = ctx.mov_f32_from_bits(c[2 * h + col]);
                                    let q = ctx.sub_f32(f, magic_f);
                                    let x = ctx.mul_f32(dsc[col], d8[2 * mt + h]);
                                    let e = ((mt * N_TILES as usize + t) * 4) + 2 * h + col;
                                    ctx.fma_f32_inplace(acc[e], x, q);
                                    ctx.fma_f32_inplace(acc[e], ndm[col], dsa[2 * mt + h]);
                                }
                            }
                        }
                    }
                }
                ctx.bar_sync(0);

                ctx.add_u32_inplace(sb, 1);
                ctx.branch("mma_sb_loop");
                ctx.label("mma_sb_end");

                // C3: store in-range elements.
                let tig2 = ctx.shl_u32_imm(tig, 1);
                for mt in 0..M_TILES {
                    for t in 0..N_TILES {
                        for h in 0..2u32 {
                            let r = ctx.mul_u32(wm, 16 * M_TILES);
                            let r = ctx.add_u32_reg(r, bm);
                            let r = ctx.add_u32_reg(r, gid);
                            let r = ctx.add_u32(r, mt * 16 + 8 * h);
                            for col in 0..2u32 {
                                let cc = ctx.add_u32_reg(bn, cw);
                                let cc = ctx.add_u32_reg(cc, tig2);
                                let cc = ctx.add_u32(cc, t * 8 + col);
                                let skip = format!("mma_st_{mt}_{t}_{h}_{col}");
                                let r_out = ctx.setp_ge_u32(r, m_dim);
                                ctx.branch_if(r_out, &skip);
                                let c_out = ctx.setp_ge_u32(cc, n_dim);
                                ctx.branch_if(c_out, &skip);
                                let e = ctx.mul_u32_reg(r, n_dim);
                                let e = ctx.add_u32_reg(e, cc);
                                let o = ctx.mul_wide_u32(e, 4);
                                let addr = ctx.add_u64(y_ptr, o);
                                let idx = ((mt * N_TILES + t) * 4 + 2 * h + col) as usize;
                                ctx.st_global_f32(addr, acc[idx]);
                                ctx.label(&skip);
                            }
                        }
                    }
                }
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mma_q4k_gemm_ptx_assembles_on_sm89() {
        let ptx = MmaQ4KGemmKernel::new(100, 1536, 1024).emit_ptx_for_target("sm_89");
        assert!(ptx.contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"));
        assert_eq!(MmaQ4KGemmKernel::grid(100, 1536), (12, 2));
        let Ok(v) = std::process::Command::new("ptxas")
            .arg("--version")
            .output()
        else {
            eprintln!("ptxas not found; emission checked only");
            return;
        };
        assert!(v.status.success());
        let path = std::env::temp_dir().join(format!("mma_q4k_{}.ptx", std::process::id()));
        std::fs::write(&path, &ptx).expect("write ptx");
        let res = std::process::Command::new("ptxas")
            .args(["--gpu-name", "sm_89", "-o", "/dev/null"])
            .arg(&path)
            .output()
            .expect("run ptxas");
        let _ = std::fs::remove_file(&path);
        assert!(
            res.status.success(),
            "ptxas rejected:\n{}",
            String::from_utf8_lossy(&res.stderr)
        );
    }
}
