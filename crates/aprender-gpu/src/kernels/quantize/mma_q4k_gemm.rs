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
const A_D8: u32 = A_QS + MMA_Q4K_TILE_M * A_ROW; // [j][row] f32
const A_DSA: u32 = A_D8 + 8 * MMA_Q4K_TILE_M * 4; // [j][row] f32, d8·Σa
const W_RAW: u32 = A_DSA + 8 * MMA_Q4K_TILE_M * 4; // [col] 144 B
const W_DSC: u32 = W_RAW + MMA_Q4K_TILE_N * Q4K_SUPER_BLOCK_BYTES; // [j][col] f32
const W_NDM: u32 = W_DSC + 8 * MMA_Q4K_TILE_N * 4; // [j][col] f32, −dmin·m
const SMEM_BYTES: u32 = W_NDM + 8 * MMA_Q4K_TILE_N * 4;

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
                    let s_d = ctx.shl_u32_imm(jb, 6); // j·64 rows
                    let s_d = ctx.add_u32_reg(s_d, row);
                    let s_d = ctx.shl_u32_imm(s_d, 2);
                    a_ld.push((g, s_qs, s_d));
                }
                // W raw: col tid/2 copies words (tid%2)·18 .. +18 of its super-block.
                let half = ctx.and_u32_imm(tid, 1);
                let wcol = ctx.shr_u32_imm(tid, 1);
                let w_g = {
                    let c = ctx.add_u32_reg(bn, wcol);
                    let c = ctx.min_u32(c, n_last);
                    let o = ctx.mul_wide_u32_reg(c, w_row_bytes);
                    let g = ctx.add_u64(w_ptr, o);
                    let h = ctx.mul_wide_u32(half, 72);
                    ctx.add_u64(g, h)
                };
                let w_s = {
                    let s = ctx.mul_u32(wcol, Q4K_SUPER_BLOCK_BYTES);
                    let h = ctx.mul_u32(half, 72);
                    let s = ctx.add_u32_reg(s, h);
                    ctx.add_u32(s, W_RAW)
                };
                // Scale unpack: threads < 128 own column tid, reading its header from smem.
                let own_scales = ctx.setp_lt_u32_imm(tid, MMA_Q4K_TILE_N);
                let hdr_s = {
                    let s = ctx.mul_u32(tid, Q4K_SUPER_BLOCK_BYTES);
                    ctx.add_u32(s, W_RAW)
                };
                let sc_s = ctx.shl_u32_imm(tid, 2);

                // ---- Fragment bases in smem ----
                let tig4 = ctx.shl_u32_imm(tig, 2);
                let mut a_fr = Vec::new(); // qs address of row gid, word tig
                let mut d_fr = Vec::new(); // [j=0][row gid] offset
                for mt in 0..M_TILES {
                    let r = ctx.mul_u32(wm, 16 * M_TILES);
                    let r = ctx.add_u32(r, mt * 16);
                    let r = ctx.add_u32_reg(r, gid);
                    let q = ctx.mul_u32(r, A_ROW);
                    a_fr.push(ctx.add_u32_reg(q, tig4));
                    d_fr.push(ctx.shl_u32_imm(r, 2));
                }
                let mut b_fr = Vec::new(); // qs address of col gid, word tig
                let cw = ctx.mul_u32(wn, 8 * N_TILES);
                for t in 0..N_TILES {
                    let c = ctx.add_u32(cw, t * 8);
                    let c = ctx.add_u32_reg(c, gid);
                    let o = ctx.mul_u32(c, Q4K_SUPER_BLOCK_BYTES);
                    let o = ctx.add_u32_reg(o, tig4);
                    b_fr.push(ctx.add_u32(o, W_RAW + 16));
                }
                let s_col = {
                    let t2 = ctx.shl_u32_imm(tig, 1);
                    let c = ctx.add_u32_reg(cw, t2);
                    ctx.shl_u32_imm(c, 2)
                };

                let mut acc = Vec::new();
                for _ in 0..M_TILES * N_TILES * 4 {
                    acc.push(ctx.mov_f32_imm(0.0));
                }
                let ones = ctx.mov_u32_imm(0x0101_0101);
                let nib = ctx.mov_u32_imm(0x0F0F_0F0F);
                let magic_f = ctx.mov_f32_imm(12_582_912.0);

                let sb = ctx.mov_u32_imm(0);
                ctx.label("mma_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb, num_sb);
                ctx.branch_if(sb_done, "mma_sb_end");

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
                    let s = ctx.add_u32(*s_d, A_D8);
                    ctx.st_shared_f32(s, d8);
                    let s = ctx.add_u32(*s_d, A_DSA);
                    ctx.st_shared_f32(s, dsa);
                }
                // ---- Stage W raw ----
                let w_sb = ctx.mul_wide_u32(sb, Q4K_SUPER_BLOCK_BYTES);
                let wg = ctx.add_u64(w_g, w_sb);
                for i in 0..18u32 {
                    let o = ctx.mov_u64_imm(u64::from(4 * i));
                    let a = ctx.add_u64(wg, o);
                    let v = ctx.ld_global_u32(a);
                    let s = ctx.add_u32(w_s, 4 * i);
                    ctx.st_shared_u32(s, v);
                }
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
                        let s = ctx.add_u32(sc_s, W_DSC + j * MMA_Q4K_TILE_N * 4);
                        ctx.st_shared_f32(s, dsc);
                        let s = ctx.add_u32(sc_s, W_NDM + j * MMA_Q4K_TILE_N * 4);
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
                        let lo = ctx.add_u32(a_fr[mt], j * 32);
                        let a0 = ctx.ld_shared_u32(lo);
                        let p = ctx.add_u32(lo, 8 * A_ROW);
                        let a1 = ctx.ld_shared_u32(p);
                        let p = ctx.add_u32(lo, 16);
                        let a2 = ctx.ld_shared_u32(p);
                        let p = ctx.add_u32(lo, 8 * A_ROW + 16);
                        let a3 = ctx.ld_shared_u32(p);
                        a_frag.push([a0, a1, a2, a3]);
                        for h in 0..2u32 {
                            let base = j * MMA_Q4K_TILE_M * 4 + h * 32;
                            let p = ctx.add_u32(d_fr[mt], A_D8 + base);
                            d8.push(ctx.ld_shared_f32(p));
                            let p = ctx.add_u32(d_fr[mt], A_DSA + base);
                            dsa.push(ctx.ld_shared_f32(p));
                        }
                    }
                    for t in 0..N_TILES as usize {
                        let bl = ctx.add_u32(b_fr[t], 32 * (j / 2));
                        let bh = ctx.add_u32(bl, 16);
                        let mut b = [ctx.ld_shared_u32(bl), ctx.ld_shared_u32(bh)];
                        for v in &mut b {
                            if j % 2 == 1 {
                                *v = ctx.shr_u32_imm(*v, 4);
                            }
                            *v = ctx.and_u32(*v, nib);
                        }
                        let mut dsc = Vec::new();
                        let mut ndm = Vec::new();
                        for col in 0..2u32 {
                            let o = j * MMA_Q4K_TILE_N * 4 + t as u32 * 32 + col * 4;
                            let p = ctx.add_u32(s_col, W_DSC + o);
                            dsc.push(ctx.ld_shared_f32(p));
                            let p = ctx.add_u32(s_col, W_NDM + o);
                            ndm.push(ctx.ld_shared_f32(p));
                        }
                        for mt in 0..M_TILES as usize {
                            let c = [0; 4].map(|_| ctx.mov_s32_imm(0x4B40_0000));
                            ctx.mma_sync_m16n8k32_s8_inplace(&a_frag[mt], &b, &c);
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
