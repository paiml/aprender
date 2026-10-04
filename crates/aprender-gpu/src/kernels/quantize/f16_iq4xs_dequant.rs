//! #3715: F16 and IQ4_XS → dense F32 dequantization, for the Qwen3.5 cuBLAS prefill.
//!
//! The Qwen3.5 batched prefill dequantizes each weight to F32 and runs one SGEMM. It had
//! kernels for Q4_K/Q5_K/Q6_K/Q8_0 only, so `Qwen3.5-4B-UD-Q4_K_XL` (F16 `ssm_alpha`/
//! `ssm_beta`, IQ4_XS FFN blocks) and `Qwen3.5-0.8B-IQ4_XS` refused the CUDA prefill, the
//! F2 guard rejected the GPU, and every cell ran on the CPU into the 600 s timeout.
//!
//! Both kernels write the value the per-token GEMV multiplies (`f16_gemv`,
//! `iq4_xs_gemv_warp_reduce`), so the batched and per-token paths see the same weights.
//! Grid `(N, blocks)`, block `(32, 1, 1)`; output is row-major `[N × K]`.

use crate::kernels::quantize::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxSync};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// IQ4_XS super-block: 256 values in 136 bytes (`d` f16, `scales_h` u16, `scales_l[4]`,
/// `qs[128]`).
pub const IQ4XS_BLOCK_SIZE: u32 = 256;
/// Bytes per IQ4_XS super-block.
pub const IQ4XS_BLOCK_BYTES: u32 = 136;

/// The 16 non-linear IQ4_NL levels (`quantize::iq_grids::KVALUES_IQ4NL`).
pub const KVALUES_IQ4NL: [i32; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// The codebook biased by +128 to fit a `u8`, four levels per `u32` word, little-endian.
/// The kernel selects a word with `nib >> 2` and a byte with `nib & 3`, then subtracts 128,
/// which is exact in f32, so no table has to live in device memory.
const fn packed_codebook() -> [u32; 4] {
    let mut words = [0u32; 4];
    let mut i = 0;
    while i < 16 {
        words[i / 4] |= ((KVALUES_IQ4NL[i] + 128) as u32) << (8 * (i % 4));
        i += 1;
    }
    words
}

/// F16 → F32 conversion of a row-major `[N × K]` matrix.
#[derive(Debug, Clone, Copy)]
pub struct F16DequantKernel {
    /// K (columns).
    pub k: u32,
    /// N (rows).
    pub n: u32,
}

impl F16DequantKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Launch grid: one warp per 32 columns of a row.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.n, self.k.div_ceil(32), 1)
    }

    /// Launch block — one warp.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (32, 1, 1)
    }
}

impl Kernel for F16DequantKernel {
    fn name(&self) -> &str {
        "f16_dequant_to_f32"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("f16_dequant_to_f32")
            .param(PtxType::U64, "out_ptr") // F32 [N × K]
            .param(PtxType::U64, "w_ptr") // F16 [N × K]
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let row = ctx.special_reg(PtxReg::CtaIdX);
                let blk = ctx.special_reg(PtxReg::CtaIdY);
                let l = ctx.special_reg(PtxReg::TidX);

                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let row_oob = ctx.setp_ge_u32(row, n_dim);
                ctx.branch_if(row_oob, "exit");
                let blk_k = ctx.mul_u32(blk, 32);
                let col = ctx.add_u32_reg(blk_k, l);
                let past_k = ctx.setp_ge_u32(col, k_dim);
                ctx.branch_if(past_k, "exit");

                let out_ptr = ctx.load_param_u64("out_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let row_k = ctx.mul_u32_reg(row, k_dim);
                let idx = ctx.add_u32_reg(row_k, col);

                let w_off = ctx.mul_wide_u32(idx, 2);
                let w_addr = ctx.add_u64(w_ptr, w_off);
                let h = ctx.ld_global_f16(w_addr);
                let val = ctx.cvt_f32_f16(h);

                let out_off = ctx.mul_wide_u32(idx, 4);
                let out_addr = ctx.add_u64(out_ptr, out_off);
                ctx.st_global_f32(out_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// IQ4_XS → F32 dequantization of a row-major `[N × K]` matrix.
#[derive(Debug, Clone, Copy)]
pub struct Iq4XsDequantKernel {
    /// K (columns; a multiple of 256 for every IQ4_XS tensor GGUF writes).
    pub k: u32,
    /// N (rows).
    pub n: u32,
}

impl Iq4XsDequantKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Launch grid: one warp per super-block.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.n, self.k.div_ceil(IQ4XS_BLOCK_SIZE), 1)
    }

    /// Launch block — one warp.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (32, 1, 1)
    }
}

impl Kernel for Iq4XsDequantKernel {
    fn name(&self) -> &str {
        "iq4_xs_dequant_to_f32"
    }

    fn build_ptx(&self) -> PtxKernel {
        let book = packed_codebook();
        PtxKernel::new("iq4_xs_dequant_to_f32")
            .param(PtxType::U64, "out_ptr") // F32 [N × K]
            .param(PtxType::U64, "w_ptr") // IQ4_XS [N × ceil(K/256) × 136 B]
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let row = ctx.special_reg(PtxReg::CtaIdX);
                let blk = ctx.special_reg(PtxReg::CtaIdY);
                let tid = ctx.special_reg(PtxReg::TidX);

                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let row_oob = ctx.setp_ge_u32(row, n_dim);
                ctx.branch_if(row_oob, "exit");

                let out_ptr = ctx.load_param_u64("out_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");

                // Block address: w + (row * blocks_per_row + blk) * 136.
                let k_round = ctx.add_u32(k_dim, IQ4XS_BLOCK_SIZE - 1);
                let blocks = ctx.div_u32(k_round, IQ4XS_BLOCK_SIZE);
                let row_blocks = ctx.mul_u32_reg(row, blocks);
                let blk_index = ctx.add_u32_reg(row_blocks, blk);
                let blk_off = ctx.mul_wide_u32(blk_index, IQ4XS_BLOCK_BYTES);
                let blk_addr = ctx.add_u64(w_ptr, blk_off);

                let d_f16 = ctx.ld_global_f16(blk_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let sh_addr = ctx.add_u64(blk_addr, two);
                let sh_u16 = ctx.ld_global_u16(sh_addr);
                let scales_h = ctx.cvt_u32_u16(sh_u16);

                // Thread `tid` owns element `tid` of every 32-value sub-block: byte
                // `tid & 15`, low nibble for tid < 16 and high nibble above.
                let jlow = ctx.and_u32_imm(tid, 15);
                let jhalf = ctx.shr_u32_imm(tid, 4);
                let nib_shift = ctx.shl_u32_imm(jhalf, 2);
                let jlow64 = ctx.cvt_u64_u32(jlow);
                let qs_base = ctx.add_u64(blk_addr, jlow64);

                let w0 = ctx.mov_u32_imm(book[0]);
                let w1 = ctx.mov_u32_imm(book[1]);
                let w2 = ctx.mov_u32_imm(book[2]);
                let w3 = ctx.mov_u32_imm(book[3]);
                let bias = ctx.mov_f32_imm(128.0);
                let thirty_two = ctx.mov_f32_imm(32.0);

                // Output column base: blk * 256 + tid, then + 32 per sub-block.
                let blk_col = ctx.mul_u32(blk, IQ4XS_BLOCK_SIZE);
                let col_base = ctx.add_u32_reg(blk_col, tid);
                let row_k = ctx.mul_u32_reg(row, k_dim);

                for m in 0..8u32 {
                    // ls = (scales_l[m/2] >> 4*(m%2)) & 0xf | ((scales_h >> 2m) & 3) << 4
                    let sl_off = ctx.mov_u64_imm(u64::from(4 + m / 2));
                    let sl_addr = ctx.add_u64(blk_addr, sl_off);
                    let sl_u8 = ctx.ld_global_u8(sl_addr);
                    let sl = ctx.cvt_u32_u8(sl_u8);
                    let ls_low = ctx.bfe_u32(sl, 4 * (m % 2), 4);
                    let hi2 = ctx.bfe_u32(scales_h, 2 * m, 2);
                    let ls_high = ctx.shl_u32_imm(hi2, 4);
                    let ls = ctx.or_u32(ls_low, ls_high);
                    let ls_f = ctx.cvt_f32_u32(ls);
                    let ls_c = ctx.sub_f32(ls_f, thirty_two);
                    let dl = ctx.mul_f32(d, ls_c);

                    let q_off = ctx.mov_u64_imm(u64::from(8 + 16 * m));
                    let q_addr = ctx.add_u64(qs_base, q_off);
                    let q_u8 = ctx.ld_global_u8(q_addr);
                    let q = ctx.cvt_u32_u8(q_u8);
                    let nib = ctx.bfe_u32_reg(q, nib_shift, 4);

                    // kvalue = byte (nib & 3) of word (nib >> 2), minus the +128 bias.
                    let ge8 = ctx.setp_ge_u32_imm(nib, 8);
                    let ge4 = ctx.setp_ge_u32_imm(nib, 4);
                    let ge12 = ctx.setp_ge_u32_imm(nib, 12);
                    let lo = ctx.selp_u32(ge4, w1, w0);
                    let hi = ctx.selp_u32(ge12, w3, w2);
                    let word = ctx.selp_u32(ge8, hi, lo);
                    let sub = ctx.and_u32_imm(nib, 3);
                    let byte_shift = ctx.shl_u32_imm(sub, 3);
                    let biased = ctx.bfe_u32_reg(word, byte_shift, 8);
                    let biased_f = ctx.cvt_f32_u32(biased);
                    let kv = ctx.sub_f32(biased_f, bias);
                    let val = ctx.mul_f32(dl, kv);

                    let col = ctx.add_u32(col_base, 32 * m);
                    let past_k = ctx.setp_ge_u32(col, k_dim);
                    let skip = format!("skip_{m}");
                    ctx.branch_if(past_k, &skip);
                    let out_idx = ctx.add_u32_reg(row_k, col);
                    let out_off = ctx.mul_wide_u32(out_idx, 4);
                    let out_addr = ctx.add_u64(out_ptr, out_off);
                    ctx.st_global_f32(out_addr, val);
                    ctx.label(&skip);
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
    fn codebook_packs_every_level() {
        let words = packed_codebook();
        for (i, &v) in KVALUES_IQ4NL.iter().enumerate() {
            let byte = (words[i / 4] >> (8 * (i % 4))) & 0xff;
            assert_eq!(byte as i32 - 128, v, "level {i}");
        }
    }

    #[test]
    fn dequant_ptx_shapes() {
        let f = F16DequantKernel::new(1024, 16);
        let ptx = f.emit_ptx();
        assert!(ptx.contains(".entry f16_dequant_to_f32"), "{ptx}");
        assert_eq!(f.grid(), (16, 32, 1));

        let q = Iq4XsDequantKernel::new(2048, 16);
        let ptx = q.emit_ptx();
        assert!(ptx.contains(".entry iq4_xs_dequant_to_f32"), "{ptx}");
        assert_eq!(ptx.matches("st.global.f32").count(), 8);
        assert_eq!(q.grid(), (16, 8, 1));
    }
}

/// Device parity against the CPU readers, bit for bit.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod f16_iq4xs_dequant_device_tests {
    use super::{F16DequantKernel, Iq4XsDequantKernel, KVALUES_IQ4NL};
    use crate::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use crate::kernels::Kernel;

    /// IEEE binary16 → f32 for normal numbers (the tests only write normal values).
    fn half_to_f32(h: u16) -> f32 {
        let sign = u32::from(h >> 15) << 31;
        let exp = u32::from((h >> 10) & 0x1F);
        let mant = u32::from(h & 0x3FF);
        assert!(exp != 0 && exp != 31, "test values are normal f16s");
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }

    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *seed >> 8
    }

    fn run(
        kernel: &dyn Kernel,
        grid: (u32, u32, u32),
        w: &[u8],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = CudaStream::new(&ctx).expect("stream");
        let (ptx, name) = (kernel.emit_ptx(), kernel.name().to_string());
        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("module");
        let w_buf = GpuBuffer::from_host(&ctx, w).expect("w");
        let out = GpuBuffer::<f32>::new(&ctx, n * k).expect("out");
        let mut args = [out.as_ptr(), w_buf.as_ptr(), k as u64, n as u64];
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast::<std::ffi::c_void>())
            .collect();
        let config = LaunchConfig {
            grid,
            block: (32, 1, 1),
            shared_mem: 0,
        };
        // SAFETY: the buffers are sized for the launch and the slots follow `.param` order.
        unsafe {
            stream
                .launch_kernel(&mut module, &name, &config, &mut raw)
                .expect("launch");
        }
        stream.synchronize().expect("sync");
        let mut host = vec![0f32; n * k];
        out.copy_to_host(&mut host).expect("copy");
        Some(host)
    }

    #[test]
    fn f16_dequant_is_bitwise_the_cpu_reader() {
        let (n, k) = (3usize, 1000usize);
        let mut seed = 0x3715_0016u32;
        let halves: Vec<u16> = (0..n * k)
            .map(|_| {
                0x2000 + (lcg(&mut seed) % 0x3800) as u16 | ((lcg(&mut seed) & 1) as u16) << 15
            })
            .collect();
        let bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let kern = F16DequantKernel::new(k as u32, n as u32);
        let Some(got) = run(&kern, kern.grid(), &bytes, n, k) else {
            println!("f16 dequant: no CUDA device — SKIPPED");
            return;
        };
        for (i, (&h, &g)) in halves.iter().zip(&got).enumerate() {
            assert_eq!(half_to_f32(h).to_bits(), g.to_bits(), "elem {i}");
        }
    }

    #[test]
    fn iq4_xs_dequant_is_bitwise_the_cpu_reader() {
        let (n, k) = (4usize, 1024usize);
        let nb = k / 256;
        let mut seed = 0x3715_0023u32;
        let mut bytes = vec![0u8; n * nb * 136];
        for b in bytes.iter_mut() {
            *b = lcg(&mut seed) as u8;
        }
        for blk in bytes.chunks_mut(136) {
            let d = 0x1800u16 + (lcg(&mut seed) % 0x0800) as u16;
            blk[..2].copy_from_slice(&d.to_le_bytes());
        }
        let kern = Iq4XsDequantKernel::new(k as u32, n as u32);
        let Some(got) = run(&kern, kern.grid(), &bytes, n, k) else {
            println!("iq4_xs dequant: no CUDA device — SKIPPED");
            return;
        };
        // The reference: ggml `dequantize_row_iq4_xs`.
        for (bi, blk) in bytes.chunks(136).enumerate() {
            let d = half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
            for ib in 0..8 {
                let ls = u32::from((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf)
                    | u32::from((scales_h >> (2 * ib)) & 3) << 4;
                let dl = d * (ls as f32 - 32.0);
                for j in 0..16 {
                    let q = blk[8 + 16 * ib + j];
                    let lo = dl * KVALUES_IQ4NL[usize::from(q & 0xf)] as f32;
                    let hi = dl * KVALUES_IQ4NL[usize::from(q >> 4)] as f32;
                    let base = bi * 256 + 32 * ib;
                    assert_eq!(
                        lo.to_bits(),
                        got[base + j].to_bits(),
                        "blk {bi} ib {ib} j {j}"
                    );
                    assert_eq!(
                        hi.to_bits(),
                        got[base + j + 16].to_bits(),
                        "blk {bi} ib {ib} j {}",
                        j + 16
                    );
                }
            }
        }
    }
}
