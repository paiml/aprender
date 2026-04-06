// =============================================================================
// PMAT-475: FUSED NF4 GATE + UP GEMM KERNEL
// =============================================================================
//
// Computes both gate and up projections in a single kernel launch, sharing the
// input activation load from DRAM. Input A[M×K] is loaded once per tile instead
// of twice (once for gate, once for up).
//
// Memory savings per call: M × K × 4 bytes (one full input read eliminated).
// For Qwen 1.5B (K=1536, M=2048): saves 12 MB per layer, 336 MB per forward.
//
// Contract: nf4-fused-gate-up-swiglu-v1.yaml
//   F-NF4-FFN-001: Output matches separate gate + up within 1e-4
//   F-NF4-FFN-002: Throughput >= 1.15x separate path

use super::super::nf4::nf4_register_lut_lookup;
use super::super::nf4_cpu::{NF4_BLOCK_SIZE, NF4_LUT};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

const NF4_BLOCK_SIZE_U32: u32 = NF4_BLOCK_SIZE as u32;
const NF4_BLOCK_DATA_BYTES: u32 = (NF4_BLOCK_SIZE / 2) as u32;

/// Fused NF4 Gate + Up GEMM kernel for training FFN.
///
/// Computes two NF4-quantized GEMMs sharing the same input activation:
/// - `gate[M×N] = A[M×K] @ dequant(W_gate_nf4[K×N])`
/// - `up[M×N]   = A[M×K] @ dequant(W_up_nf4[K×N])`
///
/// The input `A` is loaded from DRAM once per tile (shared between gate and up),
/// eliminating one full `M×K×4` byte DRAM read per call.
///
/// # Grid Configuration
///
/// Each thread block computes one `(row, col)` output element for BOTH gate and up.
/// Grid: `(N/tile, M/tile, 1)`, Block: `(tile*tile, 1, 1)`
#[derive(Debug, Clone)]
pub struct FusedNf4GateUpGemmKernel {
    /// Output rows (M = seq_len × batch)
    pub m: u32,
    /// Output columns per projection (N = intermediate_size)
    pub n: u32,
    /// Inner dimension (K = hidden_size, must be divisible by 64)
    pub k: u32,
    /// Tile size (default: 32)
    pub tile_size: u32,
}

impl FusedNf4GateUpGemmKernel {
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k, tile_size: 32 }
    }

    #[must_use]
    pub const fn num_blocks_per_col(&self) -> u32 {
        self.k / NF4_BLOCK_SIZE_U32
    }
}

impl Kernel for FusedNf4GateUpGemmKernel {
    fn name(&self) -> &str {
        "fused_nf4_gate_up_gemm"
    }

    fn build_ptx(&self) -> PtxKernel {
        let k = self.k;
        let tile = self.tile_size;
        let num_k_blocks = k / NF4_BLOCK_SIZE_U32;

        PtxKernel::new("fused_nf4_gate_up_gemm")
            .param(PtxType::U64, "gate_ptr") // Output gate [M × N]
            .param(PtxType::U64, "up_ptr") // Output up [M × N]
            .param(PtxType::U64, "a_ptr") // Input A [M × K] (shared)
            .param(PtxType::U64, "wg_scales_ptr") // Gate NF4 scales
            .param(PtxType::U64, "wg_data_ptr") // Gate NF4 packed nibbles
            .param(PtxType::U64, "wu_scales_ptr") // Up NF4 scales
            .param(PtxType::U64, "wu_data_ptr") // Up NF4 packed nibbles
            .param(PtxType::U32, "m_param") // M
            .param(PtxType::U32, "n_param") // N
            .param(PtxType::U32, "k_param") // K
            .shared_memory(16 * 4) // NF4 codebook LUT (16 f32)
            .build(move |ctx| {
                // Thread position within tile
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let block_x = ctx.special_reg(PtxReg::CtaIdX); // column tile
                let block_y = ctx.special_reg(PtxReg::CtaIdY); // row tile

                let local_row = ctx.div_u32(thread_id, tile);
                let local_col = ctx.rem_u32(thread_id, tile);
                let global_row = ctx.mul_u32(block_y, tile);
                let global_row = ctx.add_u32_reg(global_row, local_row);
                let global_col = ctx.mul_u32(block_x, tile);
                let global_col = ctx.add_u32_reg(global_col, local_col);

                // Bounds check
                let m_param = ctx.load_param_u32("m_param");
                let n_param = ctx.load_param_u32("n_param");
                let row_oob = ctx.setp_ge_u32(global_row, m_param);
                let col_oob = ctx.setp_ge_u32(global_col, n_param);
                ctx.branch_if(row_oob, "exit");
                ctx.branch_if(col_oob, "exit");

                let k_param = ctx.load_param_u32("k_param");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let up_ptr = ctx.load_param_u64("up_ptr");
                let wg_scales = ctx.load_param_u64("wg_scales_ptr");
                let wg_data = ctx.load_param_u64("wg_data_ptr");
                let wu_scales = ctx.load_param_u64("wu_scales_ptr");
                let wu_data = ctx.load_param_u64("wu_data_ptr");

                // NF4 codebook in registers
                let lut: [_; 16] = std::array::from_fn(|i| ctx.mov_f32_imm(NF4_LUT[i]));

                // Dual accumulators (f64 for precision, GH-561)
                let gate_acc = ctx.mov_f64_imm_zero();
                let up_acc = ctx.mov_f64_imm_zero();

                let four = ctx.mov_u32_imm(4);
                let thirty_two = ctx.mov_u32_imm(NF4_BLOCK_DATA_BYTES);
                let num_k_blocks_reg = ctx.mov_u32_imm(num_k_blocks);

                // Row base for A: a_ptr + global_row * K * 4
                let a_row_off = ctx.mul_u32_reg(global_row, k_param);
                let a_row_off = ctx.mul_wide_u32_reg(a_row_off, four);
                let a_row_base = ctx.add_u64(a_ptr, a_row_off);

                // Column bases for gate/up weight scales and data
                let col_scale_off = ctx.mul_u32_reg(global_col, num_k_blocks_reg);
                let col_scale_off = ctx.mul_wide_u32_reg(col_scale_off, four);
                let wg_scale_base = ctx.add_u64(wg_scales, col_scale_off);
                let wu_scale_base = ctx.add_u64(wu_scales, col_scale_off);

                let col_data_blocks = ctx.mul_u32_reg(global_col, num_k_blocks_reg);
                let col_data_off = ctx.mul_u32_reg(col_data_blocks, thirty_two);
                let col_data_off = ctx.cvt_u64_u32(col_data_off);
                let wg_data_base = ctx.add_u64(wg_data, col_data_off);
                let wu_data_base = ctx.add_u64(wu_data, col_data_off);

                // Loop over NF4 blocks (K/64 iterations)
                let blk_idx = ctx.mov_u32_imm(0);
                ctx.label("blk_loop");
                let blk_done = ctx.setp_ge_u32(blk_idx, num_k_blocks_reg);
                ctx.branch_if(blk_done, "blk_loop_end");

                // Load gate and up scales for this block
                let s_off = ctx.mul_wide_u32_reg(blk_idx, four);
                let gs_addr = ctx.add_u64(wg_scale_base, s_off);
                let us_addr = ctx.add_u64(wu_scale_base, s_off);
                let g_scale = ctx.ld_global_f32(gs_addr);
                let u_scale = ctx.ld_global_f32(us_addr);

                // Process 8 chunks of 8 values (64 values per NF4 block)
                let chunk = ctx.mov_u32_imm(0);
                let eight = ctx.mov_u32_imm(8);
                ctx.label("chunk_loop");
                let chunk_done = ctx.setp_ge_u32(chunk, eight);
                ctx.branch_if(chunk_done, "chunk_loop_end");

                // Load 4 bytes = 8 nibbles from gate and up NF4 data
                let blk_data_off = ctx.mul_u32_reg(blk_idx, thirty_two);
                let chunk_byte_off = ctx.mul_u32(chunk, 4);
                let byte_off = ctx.add_u32_reg(blk_data_off, chunk_byte_off);
                let byte_off_64 = ctx.cvt_u64_u32(byte_off);

                let gd_addr = ctx.add_u64(wg_data_base, byte_off_64);
                let ud_addr = ctx.add_u64(wu_data_base, byte_off_64);

                // Load 4 packed bytes from gate NF4 data
                let off1 = ctx.mov_u64_imm(1);
                let off2 = ctx.mov_u64_imm(2);
                let off3 = ctx.mov_u64_imm(3);
                let gb0_raw = ctx.ld_global_u8(gd_addr);
                let gb0 = ctx.cvt_u32_u8(gb0_raw);
                let gd1 = ctx.add_u64(gd_addr, off1);
                let gb1_raw = ctx.ld_global_u8(gd1);
                let gb1 = ctx.cvt_u32_u8(gb1_raw);
                let gd2 = ctx.add_u64(gd_addr, off2);
                let gb2_raw = ctx.ld_global_u8(gd2);
                let gb2 = ctx.cvt_u32_u8(gb2_raw);
                let gd3 = ctx.add_u64(gd_addr, off3);
                let gb3_raw = ctx.ld_global_u8(gd3);
                let gb3 = ctx.cvt_u32_u8(gb3_raw);

                // Load 4 packed bytes from up NF4 data
                let ub0_raw = ctx.ld_global_u8(ud_addr);
                let ub0 = ctx.cvt_u32_u8(ub0_raw);
                let ud1 = ctx.add_u64(ud_addr, off1);
                let ub1_raw = ctx.ld_global_u8(ud1);
                let ub1 = ctx.cvt_u32_u8(ub1_raw);
                let ud2 = ctx.add_u64(ud_addr, off2);
                let ub2_raw = ctx.ld_global_u8(ud2);
                let ub2 = ctx.cvt_u32_u8(ub2_raw);
                let ud3 = ctx.add_u64(ud_addr, off3);
                let ub3_raw = ctx.ld_global_u8(ud3);
                let ub3 = ctx.cvt_u32_u8(ub3_raw);

                let mask4 = ctx.mov_u32_imm(0xF);

                // Process 8 nibble pairs (gate + up) with shared A values
                let k_base = ctx.mul_u32(blk_idx, NF4_BLOCK_SIZE_U32);
                let chunk_offset = ctx.mul_u32(chunk, 8);
                let k_base = ctx.add_u32_reg(k_base, chunk_offset);

                // Extract 8 nibbles from gate bytes
                let gb0_hi = ctx.shr_u32_imm(gb0, 4);
                let gb1_hi = ctx.shr_u32_imm(gb1, 4);
                let gb2_hi = ctx.shr_u32_imm(gb2, 4);
                let gb3_hi = ctx.shr_u32_imm(gb3, 4);
                let nibs_g = [
                    ctx.and_u32(gb0, mask4),
                    ctx.and_u32(gb0_hi, mask4),
                    ctx.and_u32(gb1, mask4),
                    ctx.and_u32(gb1_hi, mask4),
                    ctx.and_u32(gb2, mask4),
                    ctx.and_u32(gb2_hi, mask4),
                    ctx.and_u32(gb3, mask4),
                    ctx.and_u32(gb3_hi, mask4),
                ];
                // Extract 8 nibbles from up bytes
                let ub0_hi = ctx.shr_u32_imm(ub0, 4);
                let ub1_hi = ctx.shr_u32_imm(ub1, 4);
                let ub2_hi = ctx.shr_u32_imm(ub2, 4);
                let ub3_hi = ctx.shr_u32_imm(ub3, 4);
                let nibs_u = [
                    ctx.and_u32(ub0, mask4),
                    ctx.and_u32(ub0_hi, mask4),
                    ctx.and_u32(ub1, mask4),
                    ctx.and_u32(ub1_hi, mask4),
                    ctx.and_u32(ub2, mask4),
                    ctx.and_u32(ub2_hi, mask4),
                    ctx.and_u32(ub3, mask4),
                    ctx.and_u32(ub3_hi, mask4),
                ];

                for i in 0..8u32 {
                    // Load A value (SHARED between gate and up — the whole point)
                    let k_idx = ctx.add_u32(k_base, i);
                    let a_off = ctx.mul_wide_u32_reg(k_idx, four);
                    let a_addr = ctx.add_u64(a_row_base, a_off);
                    let a_val = ctx.ld_global_f32(a_addr);

                    // Gate: dequant + FMA
                    let gv = nf4_register_lut_lookup(ctx, nibs_g[i as usize], &lut);
                    let gw = ctx.mul_f32(g_scale, gv);
                    ctx.fma_f64_acc_inplace(gate_acc, a_val, gw);

                    // Up: dequant + FMA (reuses same a_val!)
                    let uv = nf4_register_lut_lookup(ctx, nibs_u[i as usize], &lut);
                    let uw = ctx.mul_f32(u_scale, uv);
                    ctx.fma_f64_acc_inplace(up_acc, a_val, uw);
                }

                ctx.add_u32_inplace(chunk, 1);
                ctx.branch("chunk_loop");
                ctx.label("chunk_loop_end");

                ctx.add_u32_inplace(blk_idx, 1);
                ctx.branch("blk_loop");
                ctx.label("blk_loop_end");

                // Store outputs
                let out_idx = ctx.mul_u32_reg(global_row, n_param);
                let out_idx = ctx.add_u32_reg(out_idx, global_col);
                let out_off = ctx.mul_wide_u32_reg(out_idx, four);

                let gate_addr = ctx.add_u64(gate_ptr, out_off);
                let up_addr = ctx.add_u64(up_ptr, out_off);

                let g_f32 = ctx.cvt_f32_f64_rn(gate_acc);
                let u_f32 = ctx.cvt_f32_f64_rn(up_acc);

                ctx.st_global_f32(gate_addr, g_f32);
                ctx.st_global_f32(up_addr, u_f32);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_nf4_gate_up_name() {
        let k = FusedNf4GateUpGemmKernel::new(128, 8960, 1536);
        assert_eq!(k.name(), "fused_nf4_gate_up_gemm");
    }

    #[test]
    fn test_fused_nf4_gate_up_ptx_emits() {
        let k = FusedNf4GateUpGemmKernel::new(128, 8960, 1536);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("fused_nf4_gate_up_gemm"));
        assert!(ptx.contains("gate_ptr"));
        assert!(ptx.contains("up_ptr"));
        assert!(ptx.contains("selp")); // NF4 LUT
    }

    #[test]
    fn test_num_blocks() {
        let k = FusedNf4GateUpGemmKernel::new(128, 8960, 1536);
        assert_eq!(k.num_blocks_per_col(), 24); // 1536/64
    }
}
