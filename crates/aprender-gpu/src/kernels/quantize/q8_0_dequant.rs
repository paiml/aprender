//! PMAT-3596: Q8_0 → dense F32 dequantization, for cuBLAS SGEMM prefill.
//!
//! The Qwen3.5 GGUFs ship `ssm_alpha` and `ssm_beta` as Q8_0. A block is 34 bytes for
//! 32 values — an f16 scale `d` then 32 int8 quants — and `value = d * q`, which is
//! exactly one multiply of two exactly-converted operands, so this kernel reproduces
//! `aprender-serve`'s `dequantize_q8_0` bit for bit.
//!
//! Grid `(N, ceil(K/32))`, block `(32, 1, 1)`: one warp per block, thread `l` writes
//! value `l`. Output is row-major `[N × K]`.

use crate::kernels::quantize::{Kernel, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q8_0 → F32 dequantization.
#[derive(Debug, Clone, Copy)]
pub struct Q8_0DequantKernel {
    /// K (columns; a multiple of 32 for every Q8_0 tensor GGUF writes).
    pub k: u32,
    /// N (rows).
    pub n: u32,
}

impl Q8_0DequantKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Launch grid.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.n, self.k.div_ceil(Q8_0_BLOCK_SIZE), 1)
    }

    /// Launch block — one warp.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (32, 1, 1)
    }
}

impl Kernel for Q8_0DequantKernel {
    fn name(&self) -> &str {
        "q8_0_dequant_to_f32"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q8_0_dequant_to_f32")
            .param(PtxType::U64, "out_ptr") // F32 [N × K]
            .param(PtxType::U64, "w_ptr") // Q8_0 [N × ceil(K/32) × 34 B]
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
                let blk_k = ctx.mul_u32(blk, Q8_0_BLOCK_SIZE);
                let col = ctx.add_u32_reg(blk_k, l);
                let past_k = ctx.setp_ge_u32(col, k_dim);
                ctx.branch_if(past_k, "exit");

                let out_ptr = ctx.load_param_u64("out_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");

                // Block address: w + (row * blocks_per_row + blk) * 34.
                let k_round = ctx.add_u32(k_dim, Q8_0_BLOCK_SIZE - 1);
                let blocks = ctx.div_u32(k_round, Q8_0_BLOCK_SIZE);
                let row_blocks = ctx.mul_u32_reg(row, blocks);
                let blk_index = ctx.add_u32_reg(row_blocks, blk);
                let blk_off = ctx.mul_wide_u32(blk_index, Q8_0_BLOCK_BYTES);
                let blk_addr = ctx.add_u64(w_ptr, blk_off);

                let d_f16 = ctx.ld_global_f16(blk_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let qs = ctx.add_u64(blk_addr, two);
                let l64 = ctx.cvt_u64_u32(l);
                let q_addr = ctx.add_u64(qs, l64);
                let q_u8 = ctx.ld_global_u8(q_addr);
                let q_s32 = ctx.cvt_s32_s8(q_u8);
                let q_f = ctx.cvt_f32_s32(q_s32);
                let val = ctx.mul_f32(d, q_f);

                let row_k = ctx.mul_u32_reg(row, k_dim);
                let out_idx = ctx.add_u32_reg(row_k, col);
                let out_off = ctx.mul_wide_u32(out_idx, 4);
                let out_addr = ctx.add_u64(out_ptr, out_off);
                ctx.st_global_f32(out_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_dequant_ptx_shape() {
        let k = Q8_0DequantKernel::new(4096, 32);
        let ptx = k.emit_ptx();
        assert!(ptx.contains(".entry q8_0_dequant_to_f32"), "{ptx}");
        assert_eq!(ptx.matches("st.global.f32").count(), 1);
        assert_eq!(k.grid(), (32, 128, 1));
    }
}

/// Device parity against `aprender-serve`'s `dequantize_q8_0`, bit for bit.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod q8_0_dequant_device_tests {
    use super::Q8_0DequantKernel;
    use crate::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use crate::kernels::Kernel;

    /// IEEE binary16 → f32 for normal numbers (the test only writes normal scales).
    fn half_to_f32(h: u16) -> f32 {
        let sign = u32::from(h >> 15) << 31;
        let exp = u32::from((h >> 10) & 0x1F);
        let mant = u32::from(h & 0x3FF);
        assert!(exp != 0 && exp != 31, "test scales are normal f16s");
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }

    #[test]
    fn q8_0_dequant_is_bitwise_the_cpu_reader() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("q8_0 dequant: no CUDA device — SKIPPED");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let (n, k) = (5usize, 4096usize);
        let blocks = n * k / 32;
        let mut seed = 0x3596_0007u32;
        let mut bytes = vec![0u8; blocks * 34];
        for b in bytes.iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (seed >> 24) as u8;
        }
        for i in 0..blocks {
            let scale = 0x2000u16 + (i as u16 % 0x0C00); // normal, positive
            bytes[i * 34..i * 34 + 2].copy_from_slice(&scale.to_le_bytes());
        }
        // `dequantize_q8_0`: scale * f32::from(i8).
        let want: Vec<f32> = bytes
            .chunks(34)
            .flat_map(|b| {
                let d = half_to_f32(u16::from_le_bytes([b[0], b[1]]));
                b[2..34]
                    .iter()
                    .map(move |&q| d * f32::from(i8::from_le_bytes([q])))
                    .collect::<Vec<_>>()
            })
            .collect();

        let kernel = Q8_0DequantKernel::new(k as u32, n as u32);
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
        // SAFETY: `w` holds n*k/32 whole blocks, `out` n*k floats; u32 scalars in the
        // low half of their slots.
        unsafe {
            stream
                .launch_kernel(&mut module, kernel.name(), &config, &mut raw)
                .expect("launch");
        }
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32; n * k];
        out.copy_to_host(&mut got).expect("out");
        assert!(
            want.iter().any(|v| *v < 0.0),
            "the fixture never exercises a negative quant"
        );
        let first = got
            .iter()
            .zip(&want)
            .position(|(g, w)| g.to_bits() != w.to_bits());
        assert!(
            first.is_none(),
            "q8_0 dequant index {i}: gpu {g:e} vs cpu {w:e}",
            i = first.unwrap_or(0),
            g = got[first.unwrap_or(0)],
            w = want[first.unwrap_or(0)]
        );
    }
}
