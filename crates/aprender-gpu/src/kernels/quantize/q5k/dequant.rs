//! PMAT-3596: Q5_K → dense F32 dequantization, for cuBLAS SGEMM prefill.
//!
//! The Qwen3.5 GGUFs ship `attn_qkv` and `ssm_out` as Q5_K. The batched prefill runs
//! every projection as `dequant → f32 scratch → SGEMM`, and until this kernel only Q4_K
//! and Q6_K had a dequant leg (`Q4KDequantKernel`, `Q6KDequantKernel`).
//!
//! Block layout (ggml `block_q5_K`, 176 bytes / 256 values — the order gguf-py reads and
//! `aprender-serve`'s `for_each_q5k_value` is pinned to by FALSIFY-QDOT-007):
//!
//! ```text
//! [0..2)   d     f16
//! [2..4)   dmin  f16
//! [4..16)  scales, 6-bit scale/min pairs packed as in Q4_K (get_scale_min_k4)
//! [16..48) qh    the fifth bit: bit s of qh[l] belongs to value l of sub-block s
//! [48..176) qs   sub-blocks 2c and 2c+1 share qs[32c..32c+32]: low nibble, high nibble
//! value(s, l) = d * scale[s] * (nibble | fifth_bit << 4) - dmin * min[s]
//! ```
//!
//! Grid `(N, ceil(K/256))`, block `(32, 1, 1)`: one warp per super-block, thread `l`
//! writes value `l` of each of the eight sub-blocks. Output is row-major `[N × K]`.

use crate::kernels::quantize::{Kernel, Q5K_SUPER_BLOCK_BYTES, Q5K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q5_K → F32 dequantization.
#[derive(Debug, Clone, Copy)]
pub struct Q5KDequantKernel {
    /// K (columns; a multiple of 256 for every Q5_K tensor GGUF writes).
    pub k: u32,
    /// N (rows).
    pub n: u32,
}

impl Q5KDequantKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Super-blocks per row.
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        self.k.div_ceil(Q5K_SUPER_BLOCK_SIZE)
    }

    /// Launch grid.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.n, self.num_super_blocks_per_row(), 1)
    }

    /// Launch block — one warp.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (32, 1, 1)
    }
}

impl Kernel for Q5KDequantKernel {
    fn name(&self) -> &str {
        "q5k_dequant_to_f32"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q5k_dequant_to_f32")
            .param(PtxType::U64, "out_ptr") // F32 [N × K]
            .param(PtxType::U64, "w_ptr") // Q5_K [N × ceil(K/256) × 176 B]
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let row = ctx.special_reg(PtxReg::CtaIdX);
                let sb = ctx.special_reg(PtxReg::CtaIdY);
                let l = ctx.special_reg(PtxReg::TidX);

                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let row_oob = ctx.setp_ge_u32(row, n_dim);
                ctx.branch_if(row_oob, "exit");
                let k_round = ctx.add_u32(k_dim, Q5K_SUPER_BLOCK_SIZE - 1);
                let num_sb = ctx.div_u32(k_round, Q5K_SUPER_BLOCK_SIZE);
                let sb_oob = ctx.setp_ge_u32(sb, num_sb);
                ctx.branch_if(sb_oob, "exit");

                let out_ptr = ctx.load_param_u64("out_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");

                // Super-block address: w + (row * num_sb + sb) * 176.
                let row_sbs = ctx.mul_u32_reg(row, num_sb);
                let sb_index = ctx.add_u32_reg(row_sbs, sb);
                let sb_off = ctx.mul_wide_u32(sb_index, Q5K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(w_ptr, sb_off);

                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // The 12 packed scale bytes.
                let s: Vec<_> = (0..12u64)
                    .map(|i| {
                        let off = ctx.mov_u64_imm(4 + i);
                        let addr = ctx.add_u64(sb_addr, off);
                        let b = ctx.ld_global_u8(addr);
                        ctx.cvt_u32_u8(b)
                    })
                    .collect();
                let m6 = ctx.mov_u32_imm(0x3F);
                let m4 = ctx.mov_u32_imm(0x0F);
                let four = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);
                // get_scale_min_k4(j): j < 4 → (s[j] & 63, s[j+4] & 63);
                // j >= 4 → ((s[j+4] & 15) | (s[j-4] >> 6) << 4, (s[j+4] >> 4) | (s[j] >> 6) << 4)
                let mut ds = Vec::with_capacity(8);
                let mut dm = Vec::with_capacity(8);
                for j in 0..8usize {
                    let (sc, mn) = if j < 4 {
                        (ctx.and_u32(s[j], m6), ctx.and_u32(s[j + 4], m6))
                    } else {
                        let lo = ctx.and_u32(s[j + 4], m4);
                        let hi_src = ctx.shr_u32(s[j - 4], six);
                        let hi = ctx.shl_u32(hi_src, four);
                        let sc = ctx.or_u32(lo, hi);
                        let mlo = ctx.shr_u32(s[j + 4], four);
                        let mhi_src = ctx.shr_u32(s[j], six);
                        let mhi = ctx.shl_u32(mhi_src, four);
                        (sc, ctx.or_u32(mlo, mhi))
                    };
                    let sc_f = ctx.cvt_f32_u32(sc);
                    let mn_f = ctx.cvt_f32_u32(mn);
                    ds.push(ctx.mul_f32(d, sc_f));
                    dm.push(ctx.mul_f32(dmin, mn_f));
                }

                // qh[l], and this super-block's output base.
                let l64 = ctx.cvt_u64_u32(l);
                let qh_base_off = ctx.mov_u64_imm(16);
                let qh_base = ctx.add_u64(sb_addr, qh_base_off);
                let qh_addr = ctx.add_u64(qh_base, l64);
                let qh_b = ctx.ld_global_u8(qh_addr);
                let qh = ctx.cvt_u32_u8(qh_b);
                let one = ctx.mov_u32_imm(1);

                let sb_k = ctx.mul_u32(sb, Q5K_SUPER_BLOCK_SIZE);
                let row_k = ctx.mul_u32_reg(row, k_dim);
                let out_k = ctx.add_u32_reg(row_k, sb_k);
                let out_off = ctx.mul_wide_u32(out_k, 4);
                let out_base = ctx.add_u64(out_ptr, out_off);

                for c in 0..4u32 {
                    // qs[32c + l]: sub-block 2c is its low nibble, 2c+1 its high nibble.
                    let qs_off = ctx.mov_u64_imm(48 + 32 * u64::from(c));
                    let qs_row = ctx.add_u64(sb_addr, qs_off);
                    let qs_addr = ctx.add_u64(qs_row, l64);
                    let qb = ctx.ld_global_u8(qs_addr);
                    let q = ctx.cvt_u32_u8(qb);
                    for half in 0..2u32 {
                        let sub = 2 * c + half;
                        let nib = if half == 0 {
                            ctx.and_u32(q, m4)
                        } else {
                            ctx.shr_u32(q, four)
                        };
                        let sub_r = ctx.mov_u32_imm(sub);
                        let hb_shift = ctx.shr_u32(qh, sub_r);
                        let hb = ctx.and_u32(hb_shift, one);
                        let hb_hi = ctx.shl_u32(hb, four);
                        let q5 = ctx.or_u32(nib, hb_hi);
                        let q5_f = ctx.cvt_f32_u32(q5);
                        let scaled = ctx.mul_f32(ds[sub as usize], q5_f);
                        let val = ctx.sub_f32(scaled, dm[sub as usize]);

                        // Column sb*256 + sub*32 + l, skipped past K.
                        let col_in_sb = ctx.mov_u32_imm(sub * 32);
                        let col_local = ctx.add_u32_reg(col_in_sb, l);
                        let col = ctx.add_u32_reg(sb_k, col_local);
                        let past_k = ctx.setp_ge_u32(col, k_dim);
                        let skip = format!("q5k_dq_skip_{sub}");
                        ctx.branch_if(past_k, &skip);
                        let col_off = ctx.mul_wide_u32(col_local, 4);
                        let out_addr = ctx.add_u64(out_base, col_off);
                        ctx.st_global_f32(out_addr, val);
                        ctx.label(&skip);
                    }
                }

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q5k_dequant_ptx_shape() {
        let k = Q5KDequantKernel::new(4096, 8192);
        let ptx = k.emit_ptx();
        assert!(ptx.contains(".entry q5k_dequant_to_f32"), "{ptx}");
        assert_eq!(
            ptx.matches("st.global.f32").count(),
            8,
            "8 values per thread"
        );
        assert_eq!(k.grid(), (8192, 16, 1));
        assert_eq!(k.block(), (32, 1, 1));
    }
}

/// Device parity against a verbatim port of `aprender-serve`'s `for_each_q5k_value`.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod q5k_dequant_device_tests {
    use super::Q5KDequantKernel;
    use crate::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use crate::kernels::Kernel;

    fn f16_bits_to_f32(b: u16) -> f32 {
        half_to_f32(b)
    }

    /// IEEE binary16 → f32, exact.
    fn half_to_f32(h: u16) -> f32 {
        let sign = u32::from(h >> 15) << 31;
        let exp = u32::from((h >> 10) & 0x1F);
        let mant = u32::from(h & 0x3FF);
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                // subnormal: normalise
                let mut e = 127 - 15 + 1;
                let mut m = mant;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | ((e as u32) << 23) | ((m & 0x3FF) << 13)
            }
        } else if exp == 31 {
            sign | 0x7F80_0000 | (mant << 13)
        } else {
            sign | ((exp + 127 - 15) << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }

    /// Verbatim port of `extract_scale_min` (get_scale_min_k4).
    fn scale_min(scales: &[u8; 12], j: usize) -> (f32, f32) {
        if j < 4 {
            (f32::from(scales[j] & 63), f32::from(scales[j + 4] & 63))
        } else {
            (
                f32::from((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)),
                f32::from((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)),
            )
        }
    }

    /// Verbatim port of `for_each_q5k_value` (aprender-serve quantize/dequant_q4k.rs).
    fn cpu_block(sb: &[u8]) -> Vec<f32> {
        let d = f16_bits_to_f32(u16::from_le_bytes([sb[0], sb[1]]));
        let dmin = f16_bits_to_f32(u16::from_le_bytes([sb[2], sb[3]]));
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&sb[4..16]);
        let qh = &sb[16..48];
        let qs = &sb[48..176];
        let mut out = vec![0.0f32; 256];
        for sub in 0..8 {
            let (scale, min) = scale_min(&scales, sub);
            let (d_scale, d_min) = (d * scale, dmin * min);
            let ql = &qs[(sub / 2) * 32..(sub / 2) * 32 + 32];
            let shift = 4 * (sub % 2);
            for (l, (&q, &h)) in ql.iter().zip(qh).enumerate() {
                let q5 = ((q >> shift) & 0x0F) | (((h >> sub) & 1) << 4);
                out[sub * 32 + l] = d_scale * f32::from(q5) - d_min;
            }
        }
        out
    }

    #[test]
    fn q5k_dequant_matches_the_cpu_reader() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("q5k dequant: no CUDA device — SKIPPED");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let (n, k) = (7usize, 512usize);
        let sbs = n * (k / 256);
        // Random bytes, with d/dmin forced to modest finite f16s (0x2c00 ≈ 0.0625).
        let mut seed = 0x3596_0006u32;
        let mut bytes = vec![0u8; sbs * 176];
        for b in bytes.iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (seed >> 24) as u8;
        }
        for s in 0..sbs {
            let o = s * 176;
            bytes[o..o + 2].copy_from_slice(&(0x2c00u16 + (s as u16 % 64)).to_le_bytes());
            bytes[o + 2..o + 4].copy_from_slice(&(0x2800u16 + (s as u16 % 32)).to_le_bytes());
        }
        let want: Vec<f32> = bytes.chunks(176).flat_map(cpu_block).collect();

        let kernel = Q5KDequantKernel::new(k as u32, n as u32);
        let w = GpuBuffer::from_host(&ctx, &bytes).expect("w");
        let out = GpuBuffer::<f32>::new(&ctx, n * k).expect("out");
        let mut module = CudaModule::from_ptx(&ctx, &kernel.emit_ptx()).expect("module");
        let config = LaunchConfig {
            grid: kernel.grid(),
            block: kernel.block(),
            shared_mem: 0,
        };
        let mut args = [out.as_ptr(), w.as_ptr(), k as u64, n as u64];
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast())
            .collect();
        // SAFETY: `w` holds n*k/256 whole super-blocks, `out` holds n*k floats, and the
        // two u32 scalars sit in the low half of their slots.
        unsafe {
            stream
                .launch_kernel(&mut module, kernel.name(), &config, &mut raw)
                .expect("launch");
        }
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32; n * k];
        out.copy_to_host(&mut got).expect("out");

        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1e-3, "the reference is ~zero");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            // One mul and one sub; ptxas may contract them into an fma, which rounds
            // once instead of twice — at most 1 ulp of the result's magnitude.
            let tol = (w.abs().max(scale * 1e-3)) * 2e-7 + 1e-9;
            assert!(
                (g - w).abs() <= tol,
                "q5k dequant index {i} (row {r}, col {c}): gpu {g:e} vs cpu {w:e}",
                r = i / k,
                c = i % k
            );
        }
    }
}
