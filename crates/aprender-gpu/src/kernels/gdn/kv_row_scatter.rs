//! aprender#4233: write one KV row into a Qwen3.5 attention cache at a
//! device-resident position.
//!
//! The eager decode step writes `k`/`v` straight into `cache[pos * row ..]` through
//! a host-computed pointer view. A captured CUDA graph freezes every kernel
//! argument, so that pointer would replay the capture token's row forever. This
//! kernel takes the row from fixed scratch and reads `pos` from a device `u32`,
//! which the host updates before each replay:
//!
//! ```text
//! dst[*pos_ptr * row + i] = src[i]    for i in 0..row
//! ```
//!
//! The cache layout is `[max_len][row]` with `row = num_kv_heads * head_dim`,
//! exactly the one `gdn_decode_attention` reads.
//!
//! Grid: `(ceil(row / 256), 1, 1)`, Block: `(256, 1, 1)`.

use crate::kernels::gdn::ELEMENTWISE_BLOCK;
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// `dst[*pos_ptr * row + i] = src[i]`.
#[derive(Debug, Clone, Copy)]
pub struct KvRowScatterIndirectKernel {
    /// Floats per cache row (`num_kv_heads * head_dim`).
    pub row: u32,
    /// #5279: write `m` slots' rows in one launch; see [`Self::slots`].
    pub slots: bool,
}

impl KvRowScatterIndirectKernel {
    /// Create the kernel.
    #[must_use]
    pub const fn new(row: u32) -> Self {
        Self { row, slots: false }
    }

    /// #5279: the scatter for `m` decode slots on a `ceil(row / 256) × m`
    /// grid: `dst_ptr` is then an `[m]` table of cache pointers, and slot `r`
    /// writes row `r` of `src` into its cache at `pos_ptr[r]`.
    #[must_use]
    pub const fn slots(mut self) -> Self {
        self.slots = true;
        self
    }

    /// Launch grid.
    #[must_use]
    pub const fn grid(&self) -> (u32, u32, u32) {
        (self.row.div_ceil(ELEMENTWISE_BLOCK), 1, 1)
    }

    /// Launch block.
    #[must_use]
    pub const fn block(&self) -> (u32, u32, u32) {
        (ELEMENTWISE_BLOCK, 1, 1)
    }
}

impl Kernel for KvRowScatterIndirectKernel {
    fn name(&self) -> &str {
        if self.slots {
            "gdn_kv_row_scatter_slots"
        } else {
            "gdn_kv_row_scatter_indirect"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        let (row, slots) = (self.row, self.slots);

        PtxKernel::new(self.name())
            .param(PtxType::U64, "src_ptr") // [row]
            .param(PtxType::U64, "dst_ptr") // [max_len][row]
            .param(PtxType::U64, "pos_ptr") // device u32 position
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let cta = ctx.special_reg(PtxReg::CtaIdX);
                let block = ctx.mov_u32_imm(ELEMENTWISE_BLOCK);
                let i = ctx.mad_lo_u32(cta, block, tid);

                let row_r = ctx.mov_u32_imm(row);
                let go = ctx.setp_lt_u32(i, row_r);
                ctx.branch_if_not(go, "gdn_kv_row_scatter_exit");

                let mut src_ptr = ctx.load_param_u64("src_ptr");
                let mut dst_ptr = ctx.load_param_u64("dst_ptr");
                let mut pos_ptr = ctx.load_param_u64("pos_ptr");
                if slots {
                    // Slot `r`: its row of `src`, its cache, its position.
                    let r = ctx.special_reg(PtxReg::CtaIdY);
                    let src_row = ctx.mul_wide_u32(r, row * 4);
                    src_ptr = ctx.add_u64(src_ptr, src_row);
                    let entry_off = ctx.mul_wide_u32(r, 8);
                    let entry = ctx.add_u64(dst_ptr, entry_off);
                    dst_ptr = ctx.ld_global_u64(entry);
                    let pos_off = ctx.mul_wide_u32(r, 4);
                    pos_ptr = ctx.add_u64(pos_ptr, pos_off);
                }
                let pos = ctx.ld_global_u32(pos_ptr);

                let four = ctx.mov_u32_imm(4);
                let src_off = ctx.mul_wide_u32_reg(i, four);
                let src_addr = ctx.add_u64(src_ptr, src_off);

                // Row base in bytes, widened before the multiply: pos * row * 4 can
                // pass 2^32 on a long context.
                let row_bytes = ctx.mov_u32_imm(row * 4);
                let row_off = ctx.mul_wide_u32_reg(pos, row_bytes);
                let row_base = ctx.add_u64(dst_ptr, row_off);
                let dst_addr = ctx.add_u64(row_base, src_off);

                let v = ctx.ld_global_f32(src_addr);
                ctx.st_global_f32(dst_addr, v);

                ctx.label("gdn_kv_row_scatter_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod ptx_tests {
    use super::*;

    #[test]
    fn gdn_kv_row_scatter_ptx_reads_the_position_on_the_device() {
        let kernel = KvRowScatterIndirectKernel::new(512);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gdn_kv_row_scatter_indirect"), "{ptx}");
        assert!(ptx.contains("ld.global.u32"), "position load:\n{ptx}");
        assert!(ptx.contains("mul.wide.u32"), "64-bit row offset:\n{ptx}");
        assert_eq!(ptx.matches("st.global.f32").count(), 1, "{ptx}");
        assert_eq!(kernel.grid(), (2, 1, 1));
    }

    #[test]
    fn gdn_kv_row_scatter_slots_reads_its_cache_from_the_table() {
        let one = KvRowScatterIndirectKernel::new(512).emit_ptx();
        let rows = KvRowScatterIndirectKernel::new(512).slots().emit_ptx();
        assert!(rows.contains(".entry gdn_kv_row_scatter_slots"), "{rows}");
        assert!(!one.contains("%ctaid.y"), "{one}");
        assert!(rows.contains("%ctaid.y"), "{rows}");
        assert!(
            rows.contains("ld.global.u64"),
            "cache pointer load:\n{rows}"
        );
        assert_eq!(rows.matches("st.global.f32").count(), 1, "{rows}");
    }
}

/// Device check: the row lands at `*pos_ptr`, and nothing else in the cache moves.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod gdn_kv_row_scatter_device_tests {
    use super::KvRowScatterIndirectKernel;
    use crate::driver::{CudaContext, CudaStream, GpuBuffer};
    use crate::kernels::gdn::test_support::{run_kernel, Lcg};

    #[test]
    fn gdn_kv_row_scatter_writes_exactly_the_row_at_the_device_position() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_kv_row_scatter: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        // Qwen3.5-0.8B: 2 kv heads x 256; a row that is not a block multiple too.
        for row in [512usize, 300] {
            let max_len = 8usize;
            let kernel = KvRowScatterIndirectKernel::new(row as u32);
            let mut rng = Lcg::new(0x4233_0001);
            let cache_host = rng.vec(max_len * row, 1.0);
            let cache = GpuBuffer::from_host(&ctx, &cache_host).expect("cache");
            let mut want = cache_host.clone();
            for pos in [0u32, 5, 7] {
                let src_host = rng.vec(row, 1.0);
                let src = GpuBuffer::from_host(&ctx, &src_host).expect("src");
                let pos_buf = GpuBuffer::from_host(&ctx, &[pos]).expect("pos");
                let mut args = [src.as_ptr(), cache.as_ptr(), pos_buf.as_ptr()];
                run_kernel(
                    &ctx,
                    &stream,
                    &kernel,
                    kernel.grid(),
                    kernel.block(),
                    &mut args,
                );
                let p = pos as usize;
                want[p * row..(p + 1) * row].copy_from_slice(&src_host);
                let mut got = vec![0.0f32; max_len * row];
                cache.copy_to_host(&mut got).expect("download");
                assert_eq!(got, want, "row {row}, pos {pos}");
            }
        }
    }

    /// FALSIFY (#5279): the slots scatter writes row `r` of `src` into slot
    /// `r`'s cache at `pos[r]`, and nothing else in any cache moves.
    #[test]
    fn gdn_kv_row_scatter_slots_write_each_slot_row_at_its_position() {
        let Ok(ctx) = CudaContext::new(0) else {
            println!("gdn_kv_row_scatter_slots: no CUDA device — SKIPPED.");
            return;
        };
        let stream = CudaStream::new(&ctx).expect("stream");
        let (row, max_len) = (300usize, 8usize);
        let positions = [3u32, 0, 7];
        let kernel = KvRowScatterIndirectKernel::new(row as u32).slots();
        let mut rng = Lcg::new(0x5279_0002);
        let mut want: Vec<Vec<f32>> = (0..positions.len())
            .map(|_| rng.vec(max_len * row, 1.0))
            .collect();
        let caches: Vec<GpuBuffer<f32>> = want
            .iter()
            .map(|c| GpuBuffer::from_host(&ctx, c).expect("cache"))
            .collect();
        let src_host = rng.vec(positions.len() * row, 1.0);
        let src = GpuBuffer::from_host(&ctx, &src_host).expect("src");
        let table: Vec<u64> = caches.iter().map(GpuBuffer::as_ptr).collect();
        let table = GpuBuffer::from_host(&ctx, &table).expect("table");
        let pos = GpuBuffer::from_host(&ctx, &positions).expect("pos");
        let mut args = [src.as_ptr(), table.as_ptr(), pos.as_ptr()];
        let (gx, _, _) = kernel.grid();
        let grid = (gx, positions.len() as u32, 1);
        run_kernel(&ctx, &stream, &kernel, grid, kernel.block(), &mut args);
        for (r, (cache, &p)) in caches.iter().zip(&positions).enumerate() {
            let p = p as usize;
            want[r][p * row..(p + 1) * row].copy_from_slice(&src_host[r * row..(r + 1) * row]);
            let mut got = vec![0.0f32; max_len * row];
            cache.copy_to_host(&mut got).expect("download");
            assert_eq!(got, want[r], "slot {r}, pos {p}");
        }
    }
}
