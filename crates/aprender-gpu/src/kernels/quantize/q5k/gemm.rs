//! Q5_K FUSED GEMM KERNEL (PARITY-116)
//!
//! `C[m × n] = A[m × k] · Wᵀ`, where `A` is row-major f32 and `W` holds `n` rows of
//! `k / 256` ggml `block_q5_K` super-blocks (layout in `dequant.rs`).
//!
//! One warp per output element: lane `l` takes value `l` of each of the eight
//! sub-blocks, so a warp covers a super-block per iteration and one reduction ends it.
//!
//! #3111: this kernel used to read `qs` at 16 and `qh` at 144 (ggml has `qh` at 16,
//! `qs` at 48), took the fifth bit from a sequential bitmask, unpacked the scales as a
//! 12-bit stride scaled by 1/63 instead of `get_scale_min_k4`, and reduced across a
//! warp whose lanes held 32 different output columns.

use super::super::{Kernel, Q5K_SUPER_BLOCK_BYTES, Q5K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q5_K quantized GEMM kernel configuration
#[derive(Debug, Clone)]
pub struct Q5KKernel {
    /// Output rows (M)
    pub m: u32,
    /// Output columns (N)
    pub n: u32,
    /// Inner dimension (K) - must be divisible by 256
    pub k: u32,
    /// Output elements per block (one warp each), at most 32
    pub tile_size: u32,
}

impl Q5KKernel {
    /// Create a new Q5_K quantized GEMM kernel
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_size: 32,
        }
    }

    /// Set output elements per block (clamped to 1..=32 at launch)
    #[must_use]
    pub const fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Get number of super-blocks per row
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        self.k / Q5K_SUPER_BLOCK_SIZE
    }

    /// Warps (output elements) per block: `tile_size` within the 1024-thread limit.
    const fn warps_per_block(&self) -> u32 {
        match self.tile_size {
            0 => 1,
            t if t > 32 => 32,
            t => t,
        }
    }

    /// Launch grid: one warp per element of `C`.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        ((self.m * self.n).div_ceil(self.warps_per_block()), 1, 1)
    }

    /// Launch block.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (32 * self.warps_per_block(), 1, 1)
    }
}

impl Kernel for Q5KKernel {
    fn name(&self) -> &str {
        "q5k_gemm_ggml"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q5k_gemm_ggml")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_quant_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_quant_ptr = ctx.load_param_u64("b_quant_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Output element of this warp; a whole warp leaves together, so the
                // full-mask shuffles below never see a missing lane.
                let warps_per_block = ctx.div_u32(ntid, 32);
                let warp_in_block = ctx.div_u32(tid, 32);
                let lane = ctx.rem_u32(tid, 32);
                let block_base = ctx.mul_u32_reg(ctaid, warps_per_block);
                let out_idx = ctx.add_u32_reg(block_base, warp_in_block);
                let total = ctx.mul_u32_reg(m_param, n_param);
                let oob = ctx.setp_ge_u32(out_idx, total);
                ctx.branch_if(oob, "exit");
                let row = ctx.div_u32_reg(out_idx, n_param);
                let col = ctx.rem_u32_reg(out_idx, n_param);

                let num_k_super_blocks = ctx.div_u32(k_param, Q5K_SUPER_BLOCK_SIZE);
                let sb_bytes = ctx.mov_u32_imm(Q5K_SUPER_BLOCK_BYTES);
                let w_row_bytes = ctx.mul_u32_reg(num_k_super_blocks, sb_bytes);
                let w_row_offset = ctx.mul_wide_u32_reg(col, w_row_bytes);
                let w_row = ctx.add_u64(b_quant_ptr, w_row_offset);
                let a_row_elems = ctx.mul_wide_u32_reg(row, k_param);
                let a_row_bytes = ctx.mul_u64(a_row_elems, 4);
                let a_row = ctx.add_u64(a_ptr, a_row_bytes);

                let acc = ctx.mov_f32_imm(0.0);
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_k_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_done");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q5K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(w_row, sb_offset);

                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                // qh[lane] (offset 16): bit s is the fifth bit of value `lane` of sub-block s.
                let lane_64 = ctx.cvt_u64_u32(lane);
                let qh_off = ctx.mov_u64_imm(16);
                let qh_base = ctx.add_u64(sb_addr, qh_off);
                let qh_addr = ctx.add_u64(qh_base, lane_64);
                let qh_byte = ctx.ld_global_u8(qh_addr);
                let qh_32 = ctx.cvt_u32_u8(qh_byte);

                // qs (offset 48): sub-blocks 2c and 2c+1 share qs[32c..32c+32].
                let qs_off = ctx.mov_u64_imm(48);
                let qs_base = ctx.add_u64(sb_addr, qs_off);
                let qs_lane = ctx.add_u64(qs_base, lane_64);

                // A[row, sb*256 + lane]; sub-block s adds s*32.
                let sb_k = ctx.mul_u32(sb_idx, Q5K_SUPER_BLOCK_SIZE);
                let a_k = ctx.add_u32_reg(sb_k, lane);
                let a_k_64 = ctx.cvt_u64_u32(a_k);
                let a_k_bytes = ctx.mul_u64(a_k_64, 4);
                let a_lane = ctx.add_u64(a_row, a_k_bytes);

                let sub_block_idx = ctx.mov_u32_imm(0);
                let eight = ctx.mov_u32_imm(8);

                ctx.label("sub_block_loop");
                let sub_done = ctx.setp_ge_u32(sub_block_idx, eight);
                ctx.branch_if(sub_done, "sub_block_done");

                // get_scale_min_k4(j = sub_block_idx):
                //   j < 4:  scale = scales[j] & 63,  min = scales[j+4] & 63
                //   j >= 4: scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
                //           min   = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
                let four = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);
                let is_simple = ctx.setp_lt_u32(sub_block_idx, four);
                let j_64 = ctx.cvt_u64_u32(sub_block_idx);
                let s_j_addr = ctx.add_u64(scales_base, j_64);
                let s_j_u8 = ctx.ld_global_u8(s_j_addr);
                let s_j = ctx.cvt_u32_u8(s_j_u8);
                let j4 = ctx.add_u32_reg(sub_block_idx, four);
                let j4_64 = ctx.cvt_u64_u32(j4);
                let s_j4_addr = ctx.add_u64(scales_base, j4_64);
                let s_j4_u8 = ctx.ld_global_u8(s_j4_addr);
                let s_j4 = ctx.cvt_u32_u8(s_j4_u8);
                let zero = ctx.mov_u32_imm(0);
                let jm4_raw = ctx.sub_u32_reg(sub_block_idx, four);
                let jm4 = ctx.selp_u32(is_simple, zero, jm4_raw);
                let jm4_64 = ctx.cvt_u64_u32(jm4);
                let s_jm4_addr = ctx.add_u64(scales_base, jm4_64);
                let s_jm4_u8 = ctx.ld_global_u8(s_jm4_addr);
                let s_jm4 = ctx.cvt_u32_u8(s_jm4_u8);

                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let scale_simple = ctx.and_u32(s_j, mask_6bit);
                let min_simple = ctx.and_u32(s_j4, mask_6bit);
                let s_j4_lo = ctx.and_u32(s_j4, mask_4bit);
                let s_jm4_hi = ctx.shr_u32(s_jm4, six);
                let s_jm4_hi_sh = ctx.shl_u32(s_jm4_hi, four);
                let scale_complex = ctx.or_u32(s_j4_lo, s_jm4_hi_sh);
                let s_j4_hi = ctx.shr_u32(s_j4, four);
                let s_j_hi = ctx.shr_u32(s_j, six);
                let s_j_hi_sh = ctx.shl_u32(s_j_hi, four);
                let min_complex = ctx.or_u32(s_j4_hi, s_j_hi_sh);
                let scale_6bit = ctx.selp_u32(is_simple, scale_simple, scale_complex);
                let min_6bit = ctx.selp_u32(is_simple, min_simple, min_complex);
                let scale_f32 = ctx.cvt_f32_u32(scale_6bit);
                let min_f32 = ctx.cvt_f32_u32(min_6bit);

                // Low nibble for even sub-blocks, high nibble for odd.
                let chunk = ctx.div_u32(sub_block_idx, 2);
                let parity = ctx.rem_u32(sub_block_idx, 2);
                let chunk_bytes = ctx.mul_u32(chunk, 32);
                let chunk_bytes_64 = ctx.cvt_u64_u32(chunk_bytes);
                let qs_addr = ctx.add_u64(qs_lane, chunk_bytes_64);
                let packed = ctx.ld_global_u8(qs_addr);
                let packed_32 = ctx.cvt_u32_u8(packed);
                let nibble_shift = ctx.mul_u32(parity, 4);
                let shifted = ctx.shr_u32(packed_32, nibble_shift);
                let ql = ctx.and_u32(shifted, mask_4bit);

                let qh_shifted = ctx.shr_u32(qh_32, sub_block_idx);
                let one = ctx.mov_u32_imm(1);
                let qh_bit = ctx.and_u32(qh_shifted, one);
                let qh_high = ctx.shl_u32(qh_bit, four);
                let quant = ctx.or_u32(ql, qh_high);

                // value = d * scale * quant - dmin * min
                let quant_f32 = ctx.cvt_f32_u32(quant);
                let d_scale = ctx.mul_f32(d, scale_f32);
                let scaled = ctx.mul_f32(d_scale, quant_f32);
                let dmin_min = ctx.mul_f32(dmin, min_f32);
                let dequant = ctx.sub_f32(scaled, dmin_min);

                let sub_elems = ctx.mul_u32(sub_block_idx, 32);
                let sub_elems_64 = ctx.cvt_u64_u32(sub_elems);
                let sub_bytes = ctx.mul_u64(sub_elems_64, 4);
                let a_addr = ctx.add_u64(a_lane, sub_bytes);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.fma_f32_inplace(acc, a_val, dequant);

                ctx.add_u32_inplace(sub_block_idx, 1);
                ctx.branch("sub_block_loop");

                ctx.label("sub_block_done");
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_done");

                let tmp16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp16);
                let tmp8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp8);
                let tmp4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp4);
                let tmp2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp2);
                let tmp1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp1);

                let one_u32 = ctx.mov_u32_imm(1);
                let lane_nonzero = ctx.setp_ge_u32(lane, one_u32);
                ctx.branch_if(lane_nonzero, "exit");

                let c_row = ctx.mul_wide_u32_reg(row, n_param);
                let col_64 = ctx.cvt_u64_u32(col);
                let c_elem = ctx.add_u64(c_row, col_64);
                let c_bytes = ctx.mul_u64(c_elem, 4);
                let c_addr = ctx.add_u64(c_ptr, c_bytes);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Device parity (#3111): `C = A · Wᵀ` against a CPU matmul over the dequant test's
/// verbatim port of `aprender-serve`'s `for_each_q5k_value` (pinned to gguf-py by
/// FALSIFY-QDOT-007).
#[cfg(test)]
#[cfg(feature = "cuda")]
mod q5k_gemm_device_tests {
    use super::Q5KKernel;
    use crate::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use crate::kernels::quantize::q5k::dequant::q5k_dequant_device_tests::cpu_block;
    use crate::kernels::Kernel;

    #[test]
    fn q5k_gemm_matches_the_cpu_matmul() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("q5k gemm: no CUDA device — SKIPPED");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        // Neither dimension a multiple of 8 or 32, so a tile or warp mapping that
        // mixes outputs cannot pass by symmetry.
        let (m, n, k) = (3usize, 5usize, 512usize);
        let sbs = n * (k / 256);
        let mut seed = 0x3111_0001u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        let mut w = vec![0u8; sbs * 176];
        for b in w.iter_mut() {
            *b = next() as u8;
        }
        // d/dmin forced to modest finite f16s (0x2c00 ≈ 0.0625), as in the dequant test.
        for s in 0..sbs {
            let o = s * 176;
            w[o..o + 2].copy_from_slice(&(0x2c00u16 + (s as u16 % 64)).to_le_bytes());
            w[o + 2..o + 4].copy_from_slice(&(0x2800u16 + (s as u16 % 32)).to_le_bytes());
        }
        let a: Vec<f32> = (0..m * k)
            .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let dense: Vec<f32> = w.chunks(176).flat_map(cpu_block).collect();
        let (mut want, mut mag) = (vec![0.0f64; m * n], vec![0.0f64; m * n]);
        for r in 0..m {
            for c in 0..n {
                for i in 0..k {
                    let p = f64::from(a[r * k + i]) * f64::from(dense[c * k + i]);
                    want[r * n + c] += p;
                    mag[r * n + c] += p.abs();
                }
            }
        }

        let kernel = Q5KKernel::new(m as u32, n as u32, k as u32);
        let a_gpu = GpuBuffer::from_host(&ctx, &a).expect("a");
        let w_gpu = GpuBuffer::from_host(&ctx, &w).expect("w");
        let c_gpu = GpuBuffer::<f32>::new(&ctx, m * n).expect("c");
        let mut module = CudaModule::from_ptx(&ctx, &kernel.emit_ptx()).expect("module");
        let config = LaunchConfig {
            grid: kernel.grid(),
            block: kernel.block(),
            shared_mem: 0,
        };
        let mut args = [
            a_gpu.as_ptr(),
            w_gpu.as_ptr(),
            c_gpu.as_ptr(),
            m as u64,
            n as u64,
            k as u64,
        ];
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|p| std::ptr::from_mut(p).cast())
            .collect();
        // SAFETY: `a` holds m*k floats, `w` n*k/256 whole super-blocks, `c` m*n floats,
        // and the three u32 scalars sit in the low half of their slots.
        unsafe {
            stream
                .launch_kernel(&mut module, kernel.name(), &config, &mut raw)
                .expect("launch");
        }
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32; m * n];
        c_gpu.copy_to_host(&mut got).expect("c");

        assert!(mag.iter().all(|&s| s > 1e-2), "the reference is ~zero");
        for i in 0..m * n {
            // f32 accumulation of k products: well inside 1e-5 of the sum of |products|.
            let tol = mag[i] * 1e-5;
            assert!(
                (f64::from(got[i]) - want[i]).abs() <= tol,
                "q5k gemm C[{r}][{c}]: gpu {g} vs cpu {w} (tol {tol:e}); all gpu {got:?}",
                r = i / n,
                c = i % n,
                g = got[i],
                w = want[i]
            );
        }
    }
}
