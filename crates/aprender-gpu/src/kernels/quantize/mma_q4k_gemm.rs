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
//! ## Layout (v1: correctness first, fragments loaded straight from global)
//!
//! ```text
//! Grid:  (ceil(N/32), ceil(M/64))      Block: 4 warps
//! Warp:  16 rows (M) × 32 cols (N) = 4 mma n-tiles of 8
//! Lane:  gid = lane/4, tig = lane%4 (PTX m16n8k32 fragment roles)
//!   A a0/a2: row gid,   k tig*4.. / 16+tig*4..     a1/a3: row gid+8
//!   B b0/b1: col gid,   k tig*4.. / 16+tig*4..
//!   C c0/c1: row gid,   col tig*2, tig*2+1         c2/c3: row gid+8
//! ```
//!
//! Q4K `qs` holds sub-blocks in pairs: bytes `32p..32p+32` carry sub-block `2p` in
//! the low nibbles and `2p+1` in the high nibbles, so a B fragment is one aligned
//! u32 load plus a mask. Out-of-range rows and columns are clamped for loads and
//! skipped for stores.

use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxSync};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Rows per warp (the mma M).
const WARP_M: u32 = 16;
/// mma n-tiles per warp.
const N_TILES: u32 = 4;
/// Columns per block (= per warp).
pub const MMA_Q4K_TILE_N: u32 = 8 * N_TILES;
/// Warps per block.
pub const MMA_Q4K_WARPS: u32 = 4;
/// Rows per block.
pub const MMA_Q4K_TILE_M: u32 = WARP_M * MMA_Q4K_WARPS;
/// Q8_1 block bytes (32 s8 + f16 d + f16 s).
const Q8_BLOCK_BYTES: u32 = 36;

/// Tensor-core Q4K×Q8_1 GEMM (#4376). Requires sm_80+ and Q8_1 activations from
/// `Q8QuantizeKernel`; K must be a multiple of 256.
///
/// # Provable Contracts
///
/// - **C1**: Per sub-block, the integer dot equals the DP4A kernel's `Σ q·a`
/// - **C2**: The min term uses the exact integer `Σ a` (dp4a + group shuffle)
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
            .build(move |ctx| {
                let block_x = ctx.special_reg(PtxReg::CtaIdX);
                let block_y = ctx.special_reg(PtxReg::CtaIdY);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane = ctx.rem_u32(thread_id, 32);
                let warp = ctx.div_u32(thread_id, 32);
                let gid = ctx.shr_u32_imm(lane, 2);
                let tig = ctx.and_u32_imm(lane, 3);

                let m_dim = ctx.load_param_u32("m_dim");
                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let q8_ptr = ctx.load_param_u64("q8_ptr");

                // Warp tile origin; a warp wholly past M has nothing to do.
                let bm = ctx.mul_u32(block_y, MMA_Q4K_TILE_M);
                let wm = ctx.mul_u32(warp, WARP_M);
                let m_base = ctx.add_u32_reg(bm, wm);
                let m_out = ctx.setp_ge_u32(m_base, m_dim);
                ctx.branch_if(m_out, "mma_exit");
                let n_base = ctx.mul_u32(block_x, MMA_Q4K_TILE_N);

                let num_sb = ctx.div_u32(k_dim, Q4K_SUPER_BLOCK_SIZE);
                let w_row_bytes = ctx.mul_u32(num_sb, Q4K_SUPER_BLOCK_BYTES);
                let q8_row_bytes = ctx.mul_u32(num_sb, 8 * Q8_BLOCK_BYTES);
                let one = ctx.mov_u32_imm(1);
                let m_last = ctx.sub_u32(m_dim, one);
                let n_last = ctx.sub_u32(n_dim, one);
                let tig4 = ctx.shl_u32_imm(tig, 2);
                let tig4_64 = ctx.cvt_u64_u32(tig4);

                // A rows (clamped): q8 row bases, with and without this lane's k offset.
                let row_base = |ctx: &mut crate::ptx::builder::KernelBuilder<'_>, off: u32| {
                    let r = ctx.add_u32_reg(m_base, gid);
                    let r = ctx.add_u32(r, off);
                    let r = ctx.min_u32(r, m_last);
                    let o = ctx.mul_wide_u32_reg(r, q8_row_bytes);
                    ctx.add_u64(q8_ptr, o)
                };
                let q8_r0 = row_base(ctx, 0);
                let q8_r1 = row_base(ctx, 8);
                let q8_r0t = ctx.add_u64(q8_r0, tig4_64);
                let q8_r1t = ctx.add_u64(q8_r1, tig4_64);

                // Per n-tile: B column (gid) base with k offset, and the two C columns'
                // super-block header bases (tig*2, tig*2+1). All clamped.
                let col_base = |ctx: &mut crate::ptx::builder::KernelBuilder<'_>,
                                lane_col: crate::ptx::VirtualReg,
                                off: u32| {
                    let c = ctx.add_u32_reg(n_base, lane_col);
                    let c = ctx.add_u32(c, off);
                    let c = ctx.min_u32(c, n_last);
                    let o = ctx.mul_wide_u32_reg(c, w_row_bytes);
                    ctx.add_u64(w_ptr, o)
                };
                let tig2 = ctx.shl_u32_imm(tig, 1);
                let mut b_base = Vec::new();
                let mut h_base = Vec::new();
                for t in 0..N_TILES {
                    let b = col_base(ctx, gid, t * 8);
                    let qs_off = ctx.mov_u64_imm(16);
                    let b = ctx.add_u64(b, qs_off);
                    b_base.push(ctx.add_u64(b, tig4_64));
                    h_base.push([col_base(ctx, tig2, t * 8), col_base(ctx, tig2, t * 8 + 1)]);
                }

                let mut acc = Vec::new();
                for _ in 0..N_TILES * 4 {
                    acc.push(ctx.mov_f32_imm(0.0));
                }
                let ones = ctx.mov_u32_imm(0x0101_0101);
                let nib = ctx.mov_u32_imm(0x0F0F_0F0F);

                let sb = ctx.mov_u32_imm(0);
                ctx.label("mma_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb, num_sb);
                ctx.branch_if(sb_done, "mma_sb_end");

                let w_sb = ctx.mul_wide_u32(sb, Q4K_SUPER_BLOCK_BYTES);
                let q8_sb = ctx.mul_wide_u32(sb, 8 * Q8_BLOCK_BYTES);

                // Super-block headers of this lane's 2 C columns per n-tile:
                // d, dmin, and the three 32-bit words of the 12 packed scale bytes.
                let mut hdr = Vec::new();
                for hb in &h_base {
                    let mut cols = Vec::new();
                    for base in hb {
                        let a = ctx.add_u64(*base, w_sb);
                        let d = ctx.ld_global_f16(a);
                        let d = ctx.cvt_f32_f16(d);
                        let c2 = ctx.mov_u64_imm(2);
                        let a2 = ctx.add_u64(a, c2);
                        let dmin = ctx.ld_global_f16(a2);
                        let dmin = ctx.cvt_f32_f16(dmin);
                        let mut w = [d; 3];
                        for (i, wi) in w.iter_mut().enumerate() {
                            let ci = ctx.mov_u64_imm(4 + 4 * i as u64);
                            let ai = ctx.add_u64(a, ci);
                            *wi = ctx.ld_global_u32(ai);
                        }
                        cols.push((d, dmin, w));
                    }
                    hdr.push(cols);
                }

                for j in 0..8u32 {
                    // A fragment + per-row d8 and exact Σa for this Q8_1 block.
                    let blk = ctx.mov_u64_imm(u64::from(j * Q8_BLOCK_BYTES));
                    let blk = ctx.add_u64(q8_sb, blk);
                    let c16 = ctx.mov_u64_imm(16);
                    let c32 = ctx.mov_u64_imm(32);
                    let mut a = [ones; 4];
                    let mut d8 = [ones; 2];
                    let mut sa = [ones; 2];
                    for (h, (rt, r)) in [(q8_r0t, q8_r0), (q8_r1t, q8_r1)].into_iter().enumerate() {
                        let lo = ctx.add_u64(rt, blk);
                        let hi = ctx.add_u64(lo, c16);
                        a[h] = ctx.ld_global_u32(lo);
                        a[h + 2] = ctx.ld_global_u32(hi);
                        let da = ctx.add_u64(r, blk);
                        let da = ctx.add_u64(da, c32);
                        let dh = ctx.ld_global_f16(da);
                        d8[h] = ctx.cvt_f32_f16(dh);
                        // `dp4a.u32.s32`: unsigned ones × signed activations. (The
                        // builder's `dp4a_s32` emits `dp4a.u32.u32`, which sums the
                        // bytes unsigned.)
                        let s = ctx.mov_u32_imm(0);
                        ctx.dp4a_u32_s32_inplace(s, ones, a[h]);
                        ctx.dp4a_u32_s32_inplace(s, ones, a[h + 2]);
                        let s = ctx.cvt_f32_s32(s);
                        let t = ctx.shfl_xor_f32(s, 1);
                        let s = ctx.add_f32(s, t);
                        let t = ctx.shfl_xor_f32(s, 2);
                        sa[h] = ctx.add_f32(s, t);
                    }
                    let a_frag = a;

                    let pair_off = ctx.mov_u64_imm(u64::from(32 * (j / 2)));
                    let pair_off = ctx.add_u64(w_sb, pair_off);
                    for t in 0..N_TILES as usize {
                        let bl = ctx.add_u64(b_base[t], pair_off);
                        let bh = ctx.add_u64(bl, c16);
                        let mut b = [ctx.ld_global_u32(bl), ctx.ld_global_u32(bh)];
                        for v in &mut b {
                            if j % 2 == 1 {
                                *v = ctx.shr_u32_imm(*v, 4);
                            }
                            *v = ctx.and_u32(*v, nib);
                        }
                        let b_frag = b;
                        let c = [0, 0, 0, 0].map(|v| ctx.mov_s32_imm(v));
                        ctx.mma_sync_m16n8k32_s8_inplace(&a_frag, &b_frag, &c);

                        // Scales of sub-block j for the two C columns.
                        for (col, (d, dmin, w)) in hdr[t].iter().enumerate() {
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
                            let dsc = ctx.mul_f32(*d, sc);
                            let dmm = ctx.mul_f32(*dmin, mn);
                            for h in 0..2 {
                                // c[2h + col] is (row gid + 8h, col tig*2 + col).
                                let q = ctx.cvt_f32_s32(c[2 * h + col]);
                                let q = ctx.mul_f32(dsc, q);
                                let ms = ctx.mul_f32(dmm, sa[h]);
                                let v = ctx.sub_f32(q, ms);
                                ctx.fma_f32_inplace(acc[t * 4 + 2 * h + col], d8[h], v);
                            }
                        }
                    }
                }

                ctx.add_u32_inplace(sb, 1);
                ctx.branch("mma_sb_loop");
                ctx.label("mma_sb_end");

                // C3: store in-range elements.
                for t in 0..N_TILES {
                    for h in 0..2u32 {
                        let r = ctx.add_u32_reg(m_base, gid);
                        let r = ctx.add_u32(r, 8 * h);
                        for col in 0..2u32 {
                            let cc = ctx.add_u32_reg(n_base, tig2);
                            let cc = ctx.add_u32(cc, t * 8 + col);
                            let skip = format!("mma_st_{t}_{h}_{col}");
                            let r_out = ctx.setp_ge_u32(r, m_dim);
                            ctx.branch_if(r_out, &skip);
                            let c_out = ctx.setp_ge_u32(cc, n_dim);
                            ctx.branch_if(c_out, &skip);
                            let e = ctx.mul_u32_reg(r, n_dim);
                            let e = ctx.add_u32_reg(e, cc);
                            let o = ctx.mul_wide_u32(e, 4);
                            let addr = ctx.add_u64(y_ptr, o);
                            ctx.st_global_f32(addr, acc[(t * 4 + 2 * h + col) as usize]);
                            ctx.label(&skip);
                        }
                    }
                }

                ctx.label("mma_exit");
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
        assert_eq!(MmaQ4KGemmKernel::grid(100, 1536), (48, 2));
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
